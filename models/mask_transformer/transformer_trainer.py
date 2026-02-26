import torch
from collections import defaultdict
import torch.optim as optim
from utils.logger import create_experiment_logger
from collections import OrderedDict
from utils.utils import *
from os.path import join as pjoin
from utils.eval_t2m import evaluation_mask_transformer, evaluation_res_transformer
from models.mask_transformer.tools import *

from einops import rearrange, repeat

from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler


def def_value():
    return 0.0


# ──────────────────── Shared Base Class ────────────────────

class BaseTransformerTrainer:
    """Shared training logic for MaskTransformerTrainer and ResidualTransformerTrainer.

    Consolidates:
      - Sparse keyframe encoding (encode_frames)
      - Direction loss computation
      - Gradient clipping, AMP, NaN detection
      - Checkpoint save / resume
      - Training loop skeleton
      - Gradient statistics monitoring
    
    Subclasses must set:
      - MODEL_KEY: str   (checkpoint key, e.g. 't2m_transformer')
      - OPTIMIZER_KEY: str (checkpoint key, e.g. 'opt_t2m_transformer')
      - _model: nn.Module reference (set in subclass __init__)
    
    Subclasses must override:
      - forward(batch_data, step) -> (loss, acc)
      - _run_evaluation(...) -> best metrics tuple
    """

    MODEL_KEY = None       # Override in subclass
    OPTIMIZER_KEY = None   # Override in subclass

    def __init__(self, args, model, vq_model):
        self.opt = args
        self._model = model
        self.vq_model = vq_model
        self.device = args.device
        self.vq_model.eval()

        # Mixed precision training
        self.use_amp = getattr(args, 'use_amp', False)
        self.scaler = GradScaler() if self.use_amp else None
        if self.use_amp:
            print(f"✓ Mixed precision training (AMP) enabled for {self.__class__.__name__}")

        if args.is_train:
            self.logger = create_experiment_logger(args)

    # ── Frame encoding ─────────────────────────────────────

    def encode_frames(self, frames_batch, m_lens, target_seq_len=None):
        """
        Encode frames using SparseKeyframeEncoder (ResNet + Temporal Conv).

        Args:
            frames_batch: List of List[Path], length B
            m_lens: Tensor of motion lengths (B,) ORIGINAL lengths (before VQ downsampling)
            target_seq_len: Target sequence length to pad to (after //4)

        Returns:
            frame_embeddings: Tensor (target_seq_len, B, latent_dim)
            has_frames: bool
        """
        if not hasattr(self, 'sparse_keyframe_encoder'):
            from models.sparse_keyframe_encoder import SparseKeyframeEncoder

            resnet_arch = getattr(self.opt, 'keyframe_arch', 'resnet18')
            latent_dim = self._model.latent_dim

            print(f"Loading SparseKeyframeEncoder ({resnet_arch})...")
            self.sparse_keyframe_encoder = SparseKeyframeEncoder(
                resnet_arch=resnet_arch,
                latent_dim=latent_dim,
                pretrained=True
            ).to(self.device)
            self.sparse_keyframe_encoder.train()
            print(f"✓ SparseKeyframeEncoder loaded (arch={resnet_arch}, latent_dim={latent_dim})")

        frame_embeddings, has_frames = self.sparse_keyframe_encoder(
            frames_batch, m_lens,
            deterministic=not self.sparse_keyframe_encoder.training)

        # Pad to target sequence length if needed
        if target_seq_len is not None and frame_embeddings.shape[0] < target_seq_len:
            pad_len = target_seq_len - frame_embeddings.shape[0]
            padding = torch.zeros(pad_len, frame_embeddings.shape[1], frame_embeddings.shape[2],
                                  device=frame_embeddings.device)
            frame_embeddings = torch.cat([frame_embeddings, padding], dim=0)

        return frame_embeddings, has_frames

    # ── Batch preparation ──────────────────────────────────

    def _prepare_batch(self, batch_data, step=None):
        """Unpack batch, VQ encode, encode frames, downsample m_lens.

        Returns:
            conds, code_idx, m_lens (downsampled), frame_emb, has_frames_flag
        """
        if len(batch_data) == 4:
            conds, motion, m_lens, frames_batch = batch_data
            has_frames = True
        else:
            conds, motion, m_lens = batch_data
            frames_batch = None
            has_frames = False

        motion = motion.detach().float().to(self.device)
        m_lens = m_lens.detach().long().to(self.device)

        code_idx, _ = self.vq_model.encode(motion)

        # Encode frames BEFORE downsampling m_lens
        frame_emb = None
        has_frames_flag = False
        if has_frames and frames_batch is not None:
            target_seq_len = code_idx.shape[1]
            frame_emb, has_frames_flag = self.encode_frames(frames_batch, m_lens,
                                                            target_seq_len=target_seq_len)

        # Downsample m_lens for VQ tokens
        m_lens = torch.div(m_lens, 4, rounding_mode='floor')
        if torch.is_tensor(conds):
            # id_embedding mode: keep as LongTensor; text mode: float
            conds = conds.to(self.device)
        # else: conds is a list of strings (text mode) — leave as-is

        return conds, code_idx, m_lens, frame_emb, has_frames_flag

    # ── Direction loss ─────────────────────────────────────

    def _compute_direction_loss(self, conds, loss, device, step=None):
        """Add CLIP direction loss if fine-tuning is enabled. Returns (loss, dir_loss_value)."""
        direction_loss_value = 0.0
        if not self._model.finetune_clip:
            return loss, direction_loss_value
        # Direction loss only applies to CLIP conditioning mode
        if getattr(self._model, 'conditioning_mode', 'clip') != 'clip':
            return loss, direction_loss_value
        if not (hasattr(self.opt, 'direction_loss_weight') and self.opt.direction_loss_weight > 0):
            return loss, direction_loss_value

        compute_dir_loss = (step is None) or (step % 4 == 0)

        if compute_dir_loss:
            from utils.clip_direction_loss import contrastive_direction_loss
            with torch.set_grad_enabled(True):
                clip_embeddings = self._model.encode_text(conds)

            if torch.isnan(clip_embeddings).any():
                dir_loss = torch.tensor(0.0, device=device)
                dir_stats = {'direction_loss/total': 0.0, 'direction_loss/same': 0.0,
                             'direction_loss/opposite': 0.0}
            else:
                dir_loss, dir_stats = contrastive_direction_loss(
                    clip_embeddings, conds, temperature=0.1, margin=0.2)

            self._cached_dir_loss = dir_loss.item()
            actual_weight = min(self.opt.direction_loss_weight, 0.05)
            direction_loss_value = actual_weight * dir_loss

            if torch.isnan(direction_loss_value):
                direction_loss_value = torch.tensor(0.0, device=device)
            else:
                loss = loss + direction_loss_value

            self.last_direction_stats = dir_stats
            self.last_direction_stats['direction_loss/weighted'] = (
                direction_loss_value.item() if isinstance(direction_loss_value, torch.Tensor)
                else direction_loss_value)
            self.last_direction_stats['direction_loss/weight'] = actual_weight
        else:
            if hasattr(self, '_cached_dir_loss'):
                actual_weight = min(self.opt.direction_loss_weight, 0.05)
                direction_loss_value = actual_weight * self._cached_dir_loss

        return loss, direction_loss_value

    # ── LR warm-up ─────────────────────────────────────────

    def update_lr_warm_up(self, nb_iter, warm_up_iter, lr):
        current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
        for param_group in self._optimizer.param_groups:
            param_group["lr"] = current_lr
        return current_lr

    # ── Forward (abstract) ─────────────────────────────────

    def forward(self, batch_data, step=None):
        """Override in subclass. Returns (loss, acc)."""
        raise NotImplementedError

    # ── Update ─────────────────────────────────────────────

    def update(self, batch_data, step=None):
        self._optimizer.zero_grad()

        if self.use_amp:
            with autocast():
                loss, acc = self.forward(batch_data, step=step)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self._optimizer)
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
            self.scaler.step(self._optimizer)
            self.scaler.update()
        else:
            loss, acc = self.forward(batch_data, step=step)

            # NaN detection
            if torch.isnan(loss):
                print(f"\n{'=' * 60}")
                print(f"NaN DETECTED at iteration {step}")
                print(f"{'=' * 60}")
                if len(batch_data) >= 3:
                    motion = batch_data[1]
                    print(f"  Motion shape: {motion.shape}, has NaN: {torch.isnan(motion).any().item()}")
                raise ValueError("NaN loss detected - stopping training for debugging")

            loss.backward()

            # Gradient monitoring (every 100 iters)
            if step is not None and step % 100 == 0:
                grad_stats = self.compute_gradient_stats()
                for key, value in grad_stats.items():
                    self.logger.log_scalar(f'Gradients/{key}', value, step)

            torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
            self._optimizer.step()

        self.scheduler.step()
        return loss.item(), acc

    # ── Gradient stats ─────────────────────────────────────

    def compute_gradient_stats(self):
        """Compute gradient norm statistics for monitoring."""
        stats = {}
        transformer_grads, clip_grads, cond_emb_grads = [], [], []

        for name, param in self._model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                if 'clip_model' in name:
                    clip_grads.append(grad_norm)
                elif 'cond_emb' in name:
                    cond_emb_grads.append(grad_norm)
                else:
                    transformer_grads.append(grad_norm)

        if transformer_grads:
            stats['transformer_max'] = max(transformer_grads)
            stats['transformer_mean'] = sum(transformer_grads) / len(transformer_grads)
        if clip_grads:
            stats['clip_max'] = max(clip_grads)
            stats['clip_mean'] = sum(clip_grads) / len(clip_grads)
        if cond_emb_grads:
            stats['cond_emb_max'] = max(cond_emb_grads)
        return stats

    # ── Save / Resume ──────────────────────────────────────

    def save(self, file_name, ep, total_it):
        state_dict = self._model.state_dict()
        # Exclude CLIP weights if frozen
        if not self._model.finetune_clip:
            clip_weights = [e for e in state_dict.keys() if e.startswith('clip_model.')]
            for e in clip_weights:
                del state_dict[e]
        # Exclude frozen T5 encoder weights (always frozen)
        t5_weights = [e for e in state_dict.keys() if 'cond_provider.t5_encoder.' in e]
        for e in t5_weights:
            del state_dict[e]

        state = {
            self.MODEL_KEY: state_dict,
            self.OPTIMIZER_KEY: self._optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'ep': ep,
            'total_it': total_it,
        }

        if hasattr(self, 'sparse_keyframe_encoder'):
            state['sparse_keyframe_encoder'] = self.sparse_keyframe_encoder.state_dict()

        torch.save(state, file_name)

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)

        missing_keys, unexpected_keys = self._model.load_state_dict(
            checkpoint[self.MODEL_KEY], strict=False)

        # Filter expected missing keys
        expected_missing = []
        if not self._model.finetune_clip:
            expected_missing.extend([k for k in missing_keys if k.startswith('clip_model.')])
        # T5 encoder weights are always stripped from checkpoints
        expected_missing.extend([k for k in missing_keys if 'cond_provider.t5_encoder.' in k])
        expected_missing.extend([k for k in missing_keys if k.startswith('frame_')])
        unexpected_missing = [k for k in missing_keys if k not in expected_missing]

        if unexpected_keys:
            print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys}")
        if unexpected_missing:
            print(f"Warning: Unexpected missing keys: {unexpected_missing}")
        print(f"Loaded {self.MODEL_KEY} model")

        # Resume SparseKeyframeEncoder if available
        if 'sparse_keyframe_encoder' in checkpoint:
            if not hasattr(self, 'sparse_keyframe_encoder'):
                from models.sparse_keyframe_encoder import SparseKeyframeEncoder
                resnet_arch = getattr(self.opt, 'keyframe_arch', 'resnet18')
                latent_dim = self._model.latent_dim
                self.sparse_keyframe_encoder = SparseKeyframeEncoder(
                    resnet_arch=resnet_arch, latent_dim=latent_dim, pretrained=False
                ).to(self.device)
            self.sparse_keyframe_encoder.load_state_dict(checkpoint['sparse_keyframe_encoder'])
            print(f"Loaded SparseKeyframeEncoder")

        return checkpoint['ep'], checkpoint['total_it']

    # ── Evaluation (abstract) ──────────────────────────────

    def _run_evaluation(self, eval_val_loader, epoch, best_fid, best_div,
                        best_top1, best_top2, best_top3, best_matching,
                        eval_wrapper, plot_eval, save_ckpt, save_anim):
        """Override in subclass to call the appropriate evaluation function."""
        raise NotImplementedError

    # ── Training loop ──────────────────────────────────────

    def train(self, train_loader, val_loader, eval_val_loader, eval_wrapper, plot_eval):
        self._model.to(self.device)
        self.vq_model.to(self.device)
        self.vq_model.eval()
        for param in self.vq_model.parameters():
            param.requires_grad = False

        epoch = 0
        it = 0

        # Resume from checkpoint
        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')
            if os.path.exists(model_dir):
                print(f"Resuming training from {model_dir}")
                epoch, it = self.resume(model_dir)
                print(f"Resumed from epoch {epoch}, iteration {it}")
            else:
                print(f"Warning: --is_continue set but checkpoint not found at {model_dir}")
                print(f"Starting training from scratch...")
                self.opt.is_continue = False

        # Collect parameters
        params_to_optimize = list(self._model.parameters())
        if hasattr(self, 'sparse_keyframe_encoder'):
            params_to_optimize += list(self.sparse_keyframe_encoder.parameters())
            print(f"Optimizer: transformer + SparseKeyframeEncoder")
        else:
            print("Optimizer: transformer only")

        self._optimizer = optim.AdamW(params_to_optimize, betas=(0.9, 0.99),
                                      lr=self.opt.lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self._optimizer, milestones=self.opt.milestones, gamma=self.opt.gamma)

        # Load optimizer/scheduler state if resuming
        if self.opt.is_continue and os.path.exists(pjoin(self.opt.model_dir, 'latest.tar')):
            try:
                checkpoint = torch.load(pjoin(self.opt.model_dir, 'latest.tar'),
                                        map_location=self.device)
                if self.OPTIMIZER_KEY in checkpoint:
                    self._optimizer.load_state_dict(checkpoint[self.OPTIMIZER_KEY])
                    print("Loaded optimizer state")
                if 'scheduler' in checkpoint:
                    self.scheduler.load_state_dict(checkpoint['scheduler'])
                    print("Loaded scheduler state")
            except Exception as e:
                print(f"Warning: Could not load optimizer/scheduler state: {e}")

        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_loader)
        print(f'Total Epochs: {self.opt.max_epoch}, Total Iters: {total_iters}')
        print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_loader), len(val_loader)))
        logs = defaultdict(def_value, OrderedDict())

        # Initial evaluation
        best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer = \
            self._run_evaluation(eval_val_loader, epoch, 100, 100, 0, 0, 0, 100,
                                 eval_wrapper, plot_eval, save_ckpt=False, save_anim=False)
        best_acc = 0.
        best_loss = 100.

        while epoch < self.opt.max_epoch:
            self._model.train()
            self.vq_model.eval()
            if hasattr(self, 'sparse_keyframe_encoder'):
                self.sparse_keyframe_encoder.train()

            for i, batch in enumerate(tqdm(train_loader,
                                           desc=f"Train Epoch {epoch + 1}/{self.opt.max_epoch}")):
                it += 1
                if it < self.opt.warm_up_iter:
                    self.update_lr_warm_up(it, self.opt.warm_up_iter, self.opt.lr)

                loss, acc = self.update(batch_data=batch, step=it)
                logs['loss'] += loss
                logs['acc'] += acc
                logs['lr'] += self._optimizer.param_groups[0]['lr']

                # Log auxiliary losses if available
                for attr in ('last_direction_stats', 'last_loss_components'):
                    stats = getattr(self, attr, None)
                    if stats:
                        for key, value in stats.items():
                            if key not in logs:
                                logs[key] = 0.0
                            logs[key] += value

                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict()
                    for tag, value in logs.items():
                        self.logger.log_scalar('Train/%s' % tag, value / self.opt.log_every, it)
                        mean_loss[tag] = value / self.opt.log_every
                    logs = defaultdict(def_value, OrderedDict())
                    print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i)

                if it % self.opt.save_latest == 0:
                    self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            epoch += 1
            self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            # ── Validation ──
            print('Validation time:')
            self.vq_model.eval()
            self._model.eval()
            if hasattr(self, 'sparse_keyframe_encoder'):
                self.sparse_keyframe_encoder.eval()

            val_loss, val_acc = [], []
            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    loss, acc = self.forward(batch_data)
                    val_loss.append(loss.item())
                    val_acc.append(acc)

            mean_val_loss = np.mean(val_loss)
            mean_val_acc = np.mean(val_acc)
            print(f"Validation loss:{mean_val_loss:.3f}, accuracy:{mean_val_acc:.3f}")

            self.logger.log_scalar('Val/loss', mean_val_loss, epoch)
            self.logger.log_scalar('Val/acc', mean_val_acc, epoch)

            if mean_val_loss < best_loss:
                print(f"Improved loss from {best_loss:.02f} to {mean_val_loss}!!!")
                self.save(pjoin(self.opt.model_dir, 'net_best_loss.tar'), epoch, it)
                best_loss = mean_val_loss

            if mean_val_acc > best_acc:
                print(f"Improved accuracy from {best_acc:.02f} to {mean_val_acc}!!!")
                self.save(pjoin(self.opt.model_dir, 'net_best_acc.tar'), epoch, it)
                best_acc = mean_val_acc

            # Evaluation
            is_camera_dataset = any(name in self.opt.dataset_name.lower()
                                    for name in ["cam", "estate", "realestate"])
            
            # For id_embedding mode, still run evaluation (generates visualizations, skips text metrics)
            # For camera datasets, only evaluate periodically
            should_evaluate = not is_camera_dataset or (epoch % self.opt.eval_every_e == 0)

            if should_evaluate:
                best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer = \
                    self._run_evaluation(
                        eval_val_loader, epoch, best_fid, best_div,
                        best_top1, best_top2, best_top3, best_matching,
                        eval_wrapper, plot_eval,
                        save_ckpt=True, save_anim=(epoch % self.opt.eval_every_e == 0))
            else:
                print(f"Skipping evaluation for epoch {epoch} (camera dataset)")


# ──────────────────── Mask Transformer Trainer ────────────────────

class MaskTransformerTrainer(BaseTransformerTrainer):
    MODEL_KEY = 't2m_transformer'
    OPTIMIZER_KEY = 'opt_t2m_transformer'

    def __init__(self, args, t2m_transformer, vq_model):
        super().__init__(args, t2m_transformer, vq_model)
        # Backward-compatible attribute
        self.t2m_transformer = t2m_transformer

    def forward(self, batch_data, step=None):
        conds, code_idx, m_lens, frame_emb, has_frames_flag = self._prepare_batch(batch_data, step)

        _loss, _pred_ids, _acc = self._model(
            code_idx[..., 0], conds, m_lens,
            frame_emb=frame_emb, has_frames=has_frames_flag)

        base_recon_loss = _loss.item()

        # Direction loss
        _loss, direction_loss_value = self._compute_direction_loss(
            conds, _loss, _loss.device, step=step)

        # Store loss components for logging
        self.last_loss_components = {
            'loss/reconstruction': base_recon_loss,
            'loss/direction': (direction_loss_value.item()
                               if isinstance(direction_loss_value, torch.Tensor)
                               else direction_loss_value),
            'loss/total': _loss.item(),
        }

        return _loss, _acc

    def _run_evaluation(self, eval_val_loader, epoch, best_fid, best_div,
                        best_top1, best_top2, best_top3, best_matching,
                        eval_wrapper, plot_eval, save_ckpt, save_anim):
        return evaluation_mask_transformer(
            self.opt.save_root, eval_val_loader, self._model, self.vq_model,
            self.logger, epoch,
            best_fid=best_fid, best_div=best_div,
            best_top1=best_top1, best_top2=best_top2, best_top3=best_top3,
            best_matching=best_matching, eval_wrapper=eval_wrapper,
            plot_func=plot_eval, save_ckpt=save_ckpt, save_anim=save_anim)


# ──────────────────── Residual Transformer Trainer ────────────────────

class ResidualTransformerTrainer(BaseTransformerTrainer):
    MODEL_KEY = 'res_transformer'
    OPTIMIZER_KEY = 'opt_res_transformer'

    def __init__(self, args, res_transformer, vq_model):
        super().__init__(args, res_transformer, vq_model)
        # Backward-compatible attribute
        self.res_transformer = res_transformer

    def forward(self, batch_data, step=None):
        conds, code_idx, m_lens, frame_emb, has_frames_flag = self._prepare_batch(batch_data, step)

        ce_loss, pred_ids, acc = self._model(
            code_idx, conds, m_lens,
            frame_emb=frame_emb, has_frames=has_frames_flag)

        base_recon_loss = ce_loss.item()

        # Direction loss
        ce_loss, direction_loss_value = self._compute_direction_loss(
            conds, ce_loss, ce_loss.device, step=step)

        # Store loss components for logging
        self.last_loss_components = {
            'loss/reconstruction': base_recon_loss,
            'loss/direction': (direction_loss_value.item()
                               if isinstance(direction_loss_value, torch.Tensor)
                               else direction_loss_value),
            'loss/total': ce_loss.item(),
        }

        return ce_loss, acc

    def _run_evaluation(self, eval_val_loader, epoch, best_fid, best_div,
                        best_top1, best_top2, best_top3, best_matching,
                        eval_wrapper, plot_eval, save_ckpt, save_anim):
        return evaluation_res_transformer(
            self.opt.save_root, eval_val_loader, self._model, self.vq_model,
            self.logger, epoch,
            best_fid=best_fid, best_div=best_div,
            best_top1=best_top1, best_top2=best_top2, best_top3=best_top3,
            best_matching=best_matching, eval_wrapper=eval_wrapper,
            plot_func=plot_eval, save_ckpt=save_ckpt, save_anim=save_anim)


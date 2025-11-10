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

def def_value():
    return 0.0

class MaskTransformerTrainer:
    def __init__(self, args, t2m_transformer, vq_model):
        self.opt = args
        self.t2m_transformer = t2m_transformer
        self.vq_model = vq_model
        self.device = args.device
        self.vq_model.eval()

        if args.is_train:
            # Use enhanced logger with hyperparameter tracking
            self.logger = create_experiment_logger(args)

    def encode_frames(self, frame_paths_batch, m_lens, target_seq_len=None):
        """
        Encode frames using SparseKeyframeEncoder (ResNet + Temporal Conv).
        
        Args:
            frame_paths_batch: List of lists of Path objects, length B
                Each element is ALL frame paths for that scene (T_i frames, varies 100-280)
            m_lens: Tensor of motion lengths (B,) ORIGINAL lengths (before VQ downsampling)
            target_seq_len: Target sequence length to pad to (after //4). If None, uses max(m_lens)//4
        
        Returns:
            frame_embeddings: Tensor (target_seq_len, B, latent_dim)
                Padded to match VQ-encoded motion token sequence length
        """
        # Lazy load sparse keyframe encoder
        if not hasattr(self, 'sparse_keyframe_encoder'):
            from models.sparse_keyframe_encoder import SparseKeyframeEncoder
            
            resnet_arch = getattr(self.opt, 'keyframe_arch', 'resnet18')  # Default to resnet18
            latent_dim = self.t2m_transformer.latent_dim
            
            print(f"Loading SparseKeyframeEncoder ({resnet_arch})...")
            self.sparse_keyframe_encoder = SparseKeyframeEncoder(
                resnet_arch=resnet_arch,
                latent_dim=latent_dim,
                pretrained=True
            ).to(self.device)
            
            # Important: Enable training for ResNet fine-tuning
            self.sparse_keyframe_encoder.train()
            
            print(f"✓ SparseKeyframeEncoder loaded (arch={resnet_arch}, latent_dim={latent_dim})")
            print(f"  ResNet parameters will be fine-tuned during training")
        
        # Forward pass through sparse keyframe encoder
        # Use deterministic sampling during validation (when encoder is in eval mode)
        frame_embeddings, has_frames = self.sparse_keyframe_encoder(frame_paths_batch, m_lens, 
                                                         deterministic=not self.sparse_keyframe_encoder.training)
        
        # Pad to target sequence length if needed (to match VQ-padded motion tokens)
        if target_seq_len is not None and frame_embeddings.shape[0] < target_seq_len:
            pad_len = target_seq_len - frame_embeddings.shape[0]
            padding = torch.zeros(pad_len, frame_embeddings.shape[1], frame_embeddings.shape[2], 
                                 device=frame_embeddings.device)
            frame_embeddings = torch.cat([frame_embeddings, padding], dim=0)
            # print(f"[TRAINER DEBUG] Padded frame_emb from {frame_embeddings.shape[0] - pad_len} to {target_seq_len}")
        
        return frame_embeddings, has_frames

    def update_lr_warm_up(self, nb_iter, warm_up_iter, lr):

        current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
        for param_group in self.opt_t2m_transformer.param_groups:
            param_group["lr"] = current_lr

        return current_lr


    def forward(self, batch_data, step=None):

        # Unpack batch - handle both with/without frames
        if len(batch_data) == 4:
            conds, motion, m_lens, frame_paths = batch_data
            has_frames = True
        else:
            conds, motion, m_lens = batch_data
            frame_paths = None
            has_frames = False
        
        motion = motion.detach().float().to(self.device)
        m_lens = m_lens.detach().long().to(self.device)

        # (b, n, q)
        code_idx, _ = self.vq_model.encode(motion)
        
        # Encode frames BEFORE downsampling m_lens (need original lengths)
        frame_emb = None
        has_frames_flag = False
        if has_frames and frame_paths is not None:
            # Use ORIGINAL m_lens (motion lengths), not frame path counts
            # m_lens represents the actual trajectory length
            # Pass target_seq_len to match code_idx sequence length (after VQ padding)
            target_seq_len = code_idx.shape[1]
            frame_emb, has_frames_flag = self.encode_frames(frame_paths, m_lens, target_seq_len=target_seq_len)
        
        # Downsample m_lens for VQ tokens
        m_lens = torch.div(m_lens, 4, rounding_mode='floor')
            # frame_emb shape: (T//4, B, latent_dim) - matches motion token sequence

        conds = conds.to(self.device).float() if torch.is_tensor(conds) else conds

        # Pass frame_emb and has_frames flag to transformer
        _loss, _pred_ids, _acc = self.t2m_transformer(code_idx[..., 0], conds, m_lens, 
                                                       frame_emb=frame_emb, has_frames=has_frames_flag)

        return _loss, _acc

    def update(self, batch_data, step=None):
        loss, acc = self.forward(batch_data, step=step)

        self.opt_t2m_transformer.zero_grad()
        loss.backward()
        self.opt_t2m_transformer.step()
        self.scheduler.step()

        return loss.item(), acc

    def save(self, file_name, ep, total_it):
        t2m_trans_state_dict = self.t2m_transformer.state_dict()
        clip_weights = [e for e in t2m_trans_state_dict.keys() if e.startswith('clip_model.')]
        for e in clip_weights:
            del t2m_trans_state_dict[e]
        
        state = {
            't2m_transformer': t2m_trans_state_dict,
            'opt_t2m_transformer': self.opt_t2m_transformer.state_dict(),
            'scheduler':self.scheduler.state_dict(),
            'ep': ep,
            'total_it': total_it,
        }
        
        # Save SparseKeyframeEncoder if using frames
        if hasattr(self, 'sparse_keyframe_encoder'):
            state['sparse_keyframe_encoder'] = self.sparse_keyframe_encoder.state_dict()
            print(f"💾 Saving checkpoint with SparseKeyframeEncoder (epoch {ep})")
        
        torch.save(state, file_name)

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        missing_keys, unexpected_keys = self.t2m_transformer.load_state_dict(checkpoint['t2m_transformer'], strict=False)
        assert len(unexpected_keys) == 0
        assert all([k.startswith('clip_model.') for k in missing_keys])

        # Resume SparseKeyframeEncoder if available
        if 'sparse_keyframe_encoder' in checkpoint:
            if not hasattr(self, 'sparse_keyframe_encoder'):
                # Initialize encoder if not already created
                from models.sparse_keyframe_encoder import SparseKeyframeEncoder
                resnet_arch = getattr(self.opt, 'keyframe_arch', 'resnet18')
                latent_dim = self.t2m_transformer.latent_dim
                self.sparse_keyframe_encoder = SparseKeyframeEncoder(
                    resnet_arch=resnet_arch,
                    latent_dim=latent_dim,
                    pretrained=False  # Will load from checkpoint
                ).to(self.device)
            
            self.sparse_keyframe_encoder.load_state_dict(checkpoint['sparse_keyframe_encoder'])
            print(f"✅ Resumed SparseKeyframeEncoder from checkpoint")

        try:
            self.opt_t2m_transformer.load_state_dict(checkpoint['opt_t2m_transformer']) # Optimizer

            self.scheduler.load_state_dict(checkpoint['scheduler']) # Scheduler
        except:
            print('Resume wo optimizer')
        return checkpoint['ep'], checkpoint['total_it']


    def train(self, train_loader, val_loader, eval_val_loader, eval_wrapper, plot_eval):
        self.t2m_transformer.to(self.device)
        self.vq_model.to(self.device)

        # Collect parameters: transformer + sparse keyframe encoder (if using frames)
        params_to_optimize = list(self.t2m_transformer.parameters())
        
        if hasattr(self, 'sparse_keyframe_encoder'):
            # Add ResNet + Temporal Conv parameters for fine-tuning
            params_to_optimize += list(self.sparse_keyframe_encoder.parameters())
            print(f"Optimizer will update: transformer + SparseKeyframeEncoder ({self.opt.keyframe_arch})")
        else:
            print("Optimizer will update: transformer only")
        
        self.opt_t2m_transformer = optim.AdamW(params_to_optimize, betas=(0.9, 0.99), lr=self.opt.lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.opt_t2m_transformer,
                                                        milestones=self.opt.milestones,
                                                        gamma=self.opt.gamma)

        epoch = 0
        it = 0

        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')  # TODO
            epoch, it = self.resume(model_dir)
            print("Load model epoch:%d iterations:%d"%(epoch, it))

        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_loader)
        print(f'Total Epochs: {self.opt.max_epoch}, Total Iters: {total_iters}')
        print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_loader), len(val_loader)))
        logs = defaultdict(def_value, OrderedDict())

        best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer = evaluation_mask_transformer(
            self.opt.save_root, eval_val_loader, self.t2m_transformer, self.vq_model, self.logger, epoch,
            best_fid=100, best_div=100,
            best_top1=0, best_top2=0, best_top3=0,
            best_matching=100, eval_wrapper=eval_wrapper,
            plot_func=plot_eval, save_ckpt=False, save_anim=False
        )
        best_acc = 0.

        while epoch < self.opt.max_epoch:
            self.t2m_transformer.train()
            self.vq_model.eval()
            
            # Set frame encoder to train mode if it exists
            if hasattr(self, 'sparse_keyframe_encoder'):
                self.sparse_keyframe_encoder.train()

            for i, batch in enumerate(tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{self.opt.max_epoch}")):
                it += 1
                if it < self.opt.warm_up_iter:
                    self.update_lr_warm_up(it, self.opt.warm_up_iter, self.opt.lr)
                # where does clip encoding happen? in transformer.py. Batch contains text.
                # import pdb; pdb.set_trace()

                loss, acc = self.update(batch_data=batch)
                logs['loss'] += loss
                logs['acc'] += acc
                logs['lr'] += self.opt_t2m_transformer.param_groups[0]['lr']

                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict()
                    for tag, value in logs.items():
                        self.logger.log_scalar('Train/%s'%tag, value / self.opt.log_every, it)
                        mean_loss[tag] = value / self.opt.log_every
                    logs = defaultdict(def_value, OrderedDict())
                    print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i)

                if it % self.opt.save_latest == 0:
                    self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)
            self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)
            epoch += 1

            print('Validation time:')
            self.vq_model.eval()
            self.t2m_transformer.eval()
            
            # Set frame encoder to eval mode if it exists
            if hasattr(self, 'sparse_keyframe_encoder'):
                self.sparse_keyframe_encoder.eval()

            val_loss = []
            val_acc = []
            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    loss, acc = self.forward(batch_data)
                    val_loss.append(loss.item())
                    val_acc.append(acc)

            print(f"Validation loss:{np.mean(val_loss):.3f}, accuracy:{np.mean(val_acc):.3f}")

            self.logger.log_scalar('Val/loss', np.mean(val_loss), epoch)
            self.logger.log_scalar('Val/acc', np.mean(val_acc), epoch)

            if np.mean(val_acc) > best_acc:
                print(f"Improved accuracy from {best_acc:.02f} to {np.mean(val_acc)}!!!")
                self.save(pjoin(self.opt.model_dir, 'net_best_acc.tar'), epoch, it)
                best_acc = np.mean(val_acc)

            best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer = evaluation_mask_transformer(
                self.opt.save_root, eval_val_loader, self.t2m_transformer, self.vq_model, self.logger, epoch, best_fid=best_fid,
                best_div=best_div, best_top1=best_top1, best_top2=best_top2, best_top3=best_top3,
                best_matching=best_matching, eval_wrapper=eval_wrapper,
                plot_func=plot_eval, save_ckpt=True, save_anim=(epoch%self.opt.eval_every_e==0)
            )


class ResidualTransformerTrainer:
    def __init__(self, args, res_transformer, vq_model):
        self.opt = args
        self.res_transformer = res_transformer
        self.vq_model = vq_model
        self.device = args.device
        self.vq_model.eval()

        if args.is_train:
            # Use enhanced logger with hyperparameter tracking
            self.logger = create_experiment_logger(args)
            # self.l1_criterion = torch.nn.SmoothL1Loss()

    def encode_frames(self, frame_paths_batch, m_lens, target_seq_len=None):
        """
        Encode frames using SparseKeyframeEncoder (ResNet + Temporal Conv).
        Reuses the same encoder instance as MaskTransformerTrainer if available.
        
        Args:
            frame_paths_batch: List of lists of Path objects, length B
                Each element is ALL frame paths for that scene (T_i frames, varies 100-280)
            m_lens: Tensor of motion lengths (B,) ORIGINAL lengths (before VQ downsampling)
            target_seq_len: Target sequence length to pad to (after //4). If None, uses max(m_lens)//4
        
        Returns:
            frame_embeddings: Tensor (target_seq_len, B, latent_dim)
                Padded to match VQ-encoded motion token sequence length
        """
        # Lazy load sparse keyframe encoder
        if not hasattr(self, 'sparse_keyframe_encoder'):
            from models.sparse_keyframe_encoder import SparseKeyframeEncoder
            
            resnet_arch = getattr(self.opt, 'keyframe_arch', 'resnet18')
            latent_dim = self.res_transformer.latent_dim
            
            print(f"Loading SparseKeyframeEncoder for ResidualTransformer ({resnet_arch})...")
            self.sparse_keyframe_encoder = SparseKeyframeEncoder(
                resnet_arch=resnet_arch,
                latent_dim=latent_dim,
                pretrained=True
            ).to(self.device)
            
            # Enable training for ResNet fine-tuning
            self.sparse_keyframe_encoder.train()
            
            print(f"✓ SparseKeyframeEncoder loaded (arch={resnet_arch}, latent_dim={latent_dim})")
        
        # Forward pass through sparse keyframe encoder
        # Use deterministic sampling during validation (when encoder is in eval mode)
        frame_embeddings = self.sparse_keyframe_encoder(frame_paths_batch, m_lens,
                                                         deterministic=not self.sparse_keyframe_encoder.training)
        
        # Pad to target sequence length if needed (to match VQ-padded motion tokens)
        if target_seq_len is not None and frame_embeddings.shape[0] < target_seq_len:
            pad_len = target_seq_len - frame_embeddings.shape[0]
            padding = torch.zeros(pad_len, frame_embeddings.shape[1], frame_embeddings.shape[2], 
                                 device=frame_embeddings.device)
            frame_embeddings = torch.cat([frame_embeddings, padding], dim=0)
        
        return frame_embeddings


    def update_lr_warm_up(self, nb_iter, warm_up_iter, lr):

        current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
        for param_group in self.opt_res_transformer.param_groups:
            param_group["lr"] = current_lr

        return current_lr


    def forward(self, batch_data):
        # Unpack batch - handle both with/without frames
        if len(batch_data) == 4:
            conds, motion, m_lens, frame_paths = batch_data
            has_frames = True
        else:
            conds, motion, m_lens = batch_data
            frame_paths = None
            has_frames = False

        motion = motion.detach().float().to(self.device)
        m_lens = m_lens.detach().long().to(self.device)

        # (b, n, q), (q, b, n ,d)
        code_idx, all_codes = self.vq_model.encode(motion)
        
        # Encode frames BEFORE downsampling m_lens (need original lengths)
        frame_emb = None
        has_frames_flag = False
        if has_frames and frame_paths is not None:
            # Use ORIGINAL m_lens (motion lengths), not frame path counts
            # Pass target_seq_len to match code_idx sequence length (after VQ padding)
            target_seq_len = code_idx.shape[1]
            frame_emb, has_frames_flag = self.encode_frames(frame_paths, m_lens, target_seq_len=target_seq_len)
        
        # Now downsample m_lens to match VQ token length
        m_lens = torch.div(m_lens, 4, rounding_mode='floor')

        conds = conds.to(self.device).float() if torch.is_tensor(conds) else conds

        ce_loss, pred_ids, acc = self.res_transformer(code_idx, conds, m_lens, frame_emb=frame_emb, has_frames=has_frames_flag)

        return ce_loss, acc

    def update(self, batch_data):
        loss, acc = self.forward(batch_data)

        self.opt_res_transformer.zero_grad()
        loss.backward()
        self.opt_res_transformer.step()
        self.scheduler.step()

        return loss.item(), acc

    def save(self, file_name, ep, total_it):
        res_trans_state_dict = self.res_transformer.state_dict()
        clip_weights = [e for e in res_trans_state_dict.keys() if e.startswith('clip_model.')]
        for e in clip_weights:
            del res_trans_state_dict[e]
        
        state = {
            'res_transformer': res_trans_state_dict,
            'opt_res_transformer': self.opt_res_transformer.state_dict(),
            'scheduler':self.scheduler.state_dict(),
            'ep': ep,
            'total_it': total_it,
        }
        
        # Save SparseKeyframeEncoder if using frames
        if hasattr(self, 'sparse_keyframe_encoder'):
            state['sparse_keyframe_encoder'] = self.sparse_keyframe_encoder.state_dict()
            print(f"💾 Saving ResidualTransformer checkpoint with SparseKeyframeEncoder (epoch {ep})")
        
        torch.save(state, file_name)

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        missing_keys, unexpected_keys = self.res_transformer.load_state_dict(checkpoint['res_transformer'], strict=False)
        assert len(unexpected_keys) == 0
        assert all([k.startswith('clip_model.') for k in missing_keys])

        # Resume SparseKeyframeEncoder if available
        if 'sparse_keyframe_encoder' in checkpoint:
            if not hasattr(self, 'sparse_keyframe_encoder'):
                # Initialize encoder if not already created
                from models.sparse_keyframe_encoder import SparseKeyframeEncoder
                resnet_arch = getattr(self.opt, 'keyframe_arch', 'resnet18')
                latent_dim = self.res_transformer.latent_dim
                self.sparse_keyframe_encoder = SparseKeyframeEncoder(
                    resnet_arch=resnet_arch,
                    latent_dim=latent_dim,
                    pretrained=False  # Will load from checkpoint
                ).to(self.device)
            
            self.sparse_keyframe_encoder.load_state_dict(checkpoint['sparse_keyframe_encoder'])
            print(f"✅ Resumed SparseKeyframeEncoder for ResidualTransformer from checkpoint")

        try:
            self.opt_res_transformer.load_state_dict(checkpoint['opt_res_transformer']) # Optimizer

            self.scheduler.load_state_dict(checkpoint['scheduler']) # Scheduler
        except:
            print('Resume wo optimizer')
        return checkpoint['ep'], checkpoint['total_it']

    def train(self, train_loader, val_loader, eval_val_loader, eval_wrapper, plot_eval):
        self.res_transformer.to(self.device)
        self.vq_model.to(self.device)

        # Collect parameters: residual transformer + sparse keyframe encoder (if using frames)
        params_to_optimize = list(self.res_transformer.parameters())
        
        if hasattr(self, 'sparse_keyframe_encoder'):
            params_to_optimize += list(self.sparse_keyframe_encoder.parameters())
            print(f"ResidualTransformer optimizer: transformer + SparseKeyframeEncoder ({self.opt.keyframe_arch})")
        else:
            print("ResidualTransformer optimizer: transformer only")

        self.opt_res_transformer = optim.AdamW(params_to_optimize, betas=(0.9, 0.99), lr=self.opt.lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.opt_res_transformer,
                                                        milestones=self.opt.milestones,
                                                        gamma=self.opt.gamma)

        epoch = 0
        it = 0

        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')  # TODO
            epoch, it = self.resume(model_dir)
            print("Load model epoch:%d iterations:%d"%(epoch, it))

        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_loader)
        print(f'Total Epochs: {self.opt.max_epoch}, Total Iters: {total_iters}')
        print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_loader), len(val_loader)))
        logs = defaultdict(def_value, OrderedDict())

        best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer = evaluation_res_transformer(
            self.opt.save_root, eval_val_loader, self.res_transformer, self.vq_model, self.logger, epoch,
            best_fid=100, best_div=100,
            best_top1=0, best_top2=0, best_top3=0,
            best_matching=100, eval_wrapper=eval_wrapper,
            plot_func=plot_eval, save_ckpt=False, save_anim=False
        )
        best_loss = 100
        best_acc = 0

        while epoch < self.opt.max_epoch:
            self.res_transformer.train()
            self.vq_model.eval()
            
            # Set frame encoder to train mode if it exists
            if hasattr(self, 'sparse_keyframe_encoder'):
                self.sparse_keyframe_encoder.train()

            for i, batch in enumerate(tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{self.opt.max_epoch}")):
                it += 1
                if it < self.opt.warm_up_iter:
                    self.update_lr_warm_up(it, self.opt.warm_up_iter, self.opt.lr)

                loss, acc = self.update(batch_data=batch)
                logs['loss'] += loss
                logs["acc"] += acc
                logs['lr'] += self.opt_res_transformer.param_groups[0]['lr']

                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict()
                    for tag, value in logs.items():
                        self.logger.log_scalar('Train/%s'%tag, value / self.opt.log_every, it)
                        mean_loss[tag] = value / self.opt.log_every
                    logs = defaultdict(def_value, OrderedDict())
                    print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i)

                if it % self.opt.save_latest == 0:
                    self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            epoch += 1
            self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            print('Validation time:')
            self.vq_model.eval()
            self.res_transformer.eval()

            val_loss = []
            val_acc = []
            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    loss, acc = self.forward(batch_data)
                    val_loss.append(loss.item())
                    val_acc.append(acc)

            print(f"Validation loss:{np.mean(val_loss):.3f}, Accuracy:{np.mean(val_acc):.3f}")

            self.logger.log_scalar('Val/loss', np.mean(val_loss), epoch)
            self.logger.log_scalar('Val/acc', np.mean(val_acc), epoch)

            if np.mean(val_loss) < best_loss:
                print(f"Improved loss from {best_loss:.02f} to {np.mean(val_loss)}!!!")
                self.save(pjoin(self.opt.model_dir, 'net_best_loss.tar'), epoch, it)
                best_loss = np.mean(val_loss)

            if np.mean(val_acc) > best_acc:
                print(f"Improved acc from {best_acc:.02f} to {np.mean(val_acc)}!!!")
                # self.save(pjoin(self.opt.model_dir, 'net_best_loss.tar'), epoch, it)
                best_acc = np.mean(val_acc)

            best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer = evaluation_res_transformer(
                self.opt.save_root, eval_val_loader, self.res_transformer, self.vq_model, self.logger, epoch, best_fid=best_fid,
                best_div=best_div, best_top1=best_top1, best_top2=best_top2, best_top3=best_top3,
                best_matching=best_matching, eval_wrapper=eval_wrapper,
                plot_func=plot_eval, save_ckpt=True, save_anim=(epoch%self.opt.eval_every_e==0)
            )
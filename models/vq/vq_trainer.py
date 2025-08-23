import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.nn.utils.clip_grad import clip_grad_norm_
from os.path import join as pjoin
import torch.nn.functional as F
import torch.optim as optim

import time
import numpy as np
from collections import OrderedDict, defaultdict
from utils.eval_t2m import evaluation_vqvae
from utils.camera_eval import evaluation_camera_vqvae
from utils.utils import print_current_loss
from utils.logging_utils import VQLogger, create_loss_dict, aggregate_losses
from utils.visual_consistency import create_visual_consistency_module

import os
import sys
from tqdm import tqdm

def def_value():
    return 0.0


class RVQTokenizerTrainer:
    def __init__(self, args, vq_model):
        self.opt = args
        self.vq_model = vq_model
        self.device = args.device

        if args.is_train:
            self.logger = VQLogger(args)
            if args.recons_loss == 'l1':
                self.l1_criterion = nn.L1Loss()
            elif args.recons_loss == 'l1_smooth':
                self.l1_criterion = nn.SmoothL1Loss()

        # Initialize visual consistency module
        try:
            from utils.visual_consistency import create_visual_consistency_module
            self.visual_consistency = create_visual_consistency_module(args)
            print(f"Visual consistency module enabled: {self.visual_consistency.enabled}")
        except ImportError:
            print("Visual consistency module not available")
            self.visual_consistency = None

        # self.critic = CriticWrapper(self.opt.dataset_name, self.opt.device)

    def forward(self, batch_data, step=None):
        motions = batch_data.detach().to(self.device).float()
        pred_motion, loss_commit, perplexity = self.vq_model(motions)
        
        self.motions = motions
        self.pred_motion = pred_motion

        loss_rec = self.l1_criterion(pred_motion, motions)
        
        if self.opt.dataset_name == 'cam':
            # For camera data, use position and orientation separately
            pred_pos = pred_motion[..., :3]  # x, y, z
            gt_pos = motions[..., :3]
            pred_ori = pred_motion[..., 3:]  # pitch, yaw
            gt_ori = motions[..., 3:]
            
            # Position loss
            loss_pos = self.l1_criterion(pred_pos, gt_pos)
            # Orientation loss (with angle wrapping consideration)
            loss_ori = self.l1_criterion(pred_ori, gt_ori)
            loss_explicit = loss_pos + loss_ori
        else:
            # For human motion data, use original local position loss
            pred_local_pos = pred_motion[..., 4 : (self.opt.joints_num - 1) * 3 + 4]
            local_pos = motions[..., 4 : (self.opt.joints_num - 1) * 3 + 4]
            loss_explicit = self.l1_criterion(pred_local_pos, local_pos)

        loss = loss_rec + self.opt.loss_vel * loss_explicit + self.opt.commit * loss_commit

        # Visual consistency loss
        loss_lpips = torch.tensor(0.0, device=self.device)
        if (self.visual_consistency is not None and 
            self.visual_consistency.enabled and 
            step is not None and 
            step % getattr(self.opt, 'visual_consistency_freq', 10) == 0 and
            self.opt.dataset_name == 'cam'):
            
            batch_size = motions.shape[0]
            total_lpips_loss = 0.0
            num_valid_samples = 0
            
            for i in range(batch_size):
                pred_traj = pred_motion[i].detach()  # [T, 5]
                gt_traj = motions[i]  # [T, 5]
                
                # Compute visual consistency loss for this sample
                try:
                    visual_losses = self.visual_consistency.compute_visual_loss(
                        pred_traj, gt_traj, data_id=None, step=step)
                    total_lpips_loss += visual_losses['lpips_loss']
                    num_valid_samples += 1
                except Exception as e:
                    # Handle any errors gracefully
                    continue
            
            if num_valid_samples > 0:
                loss_lpips = total_lpips_loss / num_valid_samples
                loss += getattr(self.opt, 'visual_consistency_weight', 0.01) * loss_lpips

        # return loss, loss_rec, loss_vel, loss_commit, perplexity
        # return loss, loss_rec, loss_percept, loss_commit, perplexity
        return loss, loss_rec, loss_explicit, loss_commit, perplexity, loss_lpips
        return loss, loss_rec, loss_explicit, loss_commit, perplexity


    # @staticmethod
    def update_lr_warm_up(self, nb_iter, warm_up_iter, lr):

        current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
        for param_group in self.opt_vq_model.param_groups:
            param_group["lr"] = current_lr

        return current_lr

    def save(self, file_name, ep, total_it, metrics=None):
        state = {
            "vq_model": self.vq_model.state_dict(),
            "opt_vq_model": self.opt_vq_model.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            'ep': ep,
            'total_it': total_it,
        }
        torch.save(state, file_name)
        
        # Log checkpoint info if metrics provided
        if metrics and hasattr(self.logger, 'save_checkpoint_info'):
            self.logger.save_checkpoint_info(ep, total_it, file_name, metrics)

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        self.vq_model.load_state_dict(checkpoint['vq_model'])
        self.opt_vq_model.load_state_dict(checkpoint['opt_vq_model'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        return checkpoint['ep'], checkpoint['total_it']

    def train(self, train_loader, val_loader, eval_val_loader, eval_wrapper, plot_eval=None):
        self.vq_model.to(self.device)
        
        # Watch model for gradient/weight logging if enabled
        if hasattr(self.logger, 'watch_model'):
            self.logger.watch_model(self.vq_model)
        
        # Log dataset information
        if hasattr(self.logger, 'log_dataset_info'):
            motion_length_stats = {
                'mean': self.opt.window_size,  # This is the training window size
                'dataset': self.opt.dataset_name
            }
            self.logger.log_dataset_info(len(train_loader.dataset), len(val_loader.dataset), motion_length_stats)

        self.opt_vq_model = optim.AdamW(self.vq_model.parameters(), lr=self.opt.lr, betas=(0.9, 0.99), weight_decay=self.opt.weight_decay)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.opt_vq_model, milestones=self.opt.milestones, gamma=self.opt.gamma)

        epoch = 0
        it = 0
        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')
            epoch, it = self.resume(model_dir)
            print("Load model epoch:%d iterations:%d"%(epoch, it))

        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_loader)
        print(f'Total Epochs: {self.opt.max_epoch}, Total Iters: {total_iters}')
        print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_loader), len(val_loader))) #????
        # val_loss = 0
        # min_val_loss = np.inf
        # min_val_epoch = epoch
        current_lr = self.opt.lr
        logs = defaultdict(def_value, OrderedDict())

        # sys.exit()
        # Initialize best metrics
        best_fid = 1000
        best_div = 100
        best_top1 = 0
        best_top2 = 0
        best_top3 = 0
        best_matching = 100
        best_recon = float('inf')
        best_smoothness = float('inf')
        best_position_error = float('inf')
        best_orientation_error = float('inf')
        
        # if self.opt.eval_on:
        if self.opt.dataset_name == 'cam':
            # Use camera-specific evaluation (now includes FID and other motion metrics)
            best_recon, best_smoothness, best_position_error, best_orientation_error, writer = evaluation_camera_vqvae(
                self.opt.model_dir, eval_val_loader, self.vq_model, self.logger, epoch, 
                best_recon=best_recon, best_smoothness=best_smoothness,
                best_position_error=best_position_error, best_orientation_error=best_orientation_error,
                eval_wrapper=eval_wrapper, save=False)
        else:
            # Use original evaluation for human motion
            best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer = evaluation_vqvae(
                self.opt.model_dir, eval_val_loader, self.vq_model, self.logger, epoch, best_fid=best_fid,
                best_div=best_div, best_top1=best_top1,
                best_top2=best_top2, best_top3=best_top3, best_matching=best_matching,
                eval_wrapper=eval_wrapper, save=False)

        while epoch < self.opt.max_epoch:
            self.vq_model.train()
            for i, batch_data in tqdm(enumerate(train_loader), desc = f"[Train RVQ epoch {epoch}/{self.opt.max_epoch}]", total=len(train_loader)):
                it += 1
                if it < self.opt.warm_up_iter:
                    current_lr = self.update_lr_warm_up(it, self.opt.warm_up_iter, self.opt.lr)
                loss, loss_rec, loss_vel, loss_commit, perplexity, loss_lpips = self.forward(batch_data, step=it)
                self.opt_vq_model.zero_grad()
                loss.backward()
                self.opt_vq_model.step()

                if it >= self.opt.warm_up_iter:
                    self.scheduler.step()
                
                logs['loss'] += loss.item()
                logs['loss_rec'] += loss_rec.item()
                # Note it not necessarily velocity, too lazy to change the name now
                logs['loss_vel'] += loss_vel.item()
                logs['loss_commit'] += loss_commit.item()
                logs['perplexity'] += perplexity.item()
                logs['lr'] += self.opt_vq_model.param_groups[0]['lr']
                logs['loss_lpips'] += loss_lpips.item() if isinstance(loss_lpips, torch.Tensor) else loss_lpips

                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict()
                    # Normalize accumulated logs
                    for tag, value in logs.items():
                        mean_loss[tag] = value / self.opt.log_every
                    
                    # Log training metrics
                    lr = self.opt_vq_model.param_groups[0]['lr']
                    self.logger.log_training_metrics(mean_loss, lr, it)
                    
                    # Log additional metrics if enabled
                    if self.logger.should_log_gradients:
                        self.logger.log_gradients(self.vq_model, it)
                    
                    if self.logger.should_log_codebook_usage:
                        self.logger.log_codebook_usage(self.vq_model.quantizer, it)
                    
                    logs = defaultdict(def_value, OrderedDict())
                    print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i)

                if it % self.opt.save_latest == 0:
                    self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            # Calculate epoch metrics for checkpoint saving
            epoch_metrics = {
                'epoch': epoch,
                'iteration': it,
                'train_loss': logs.get('loss', 0) / max(1, len(train_loader)),
                'lr': self.opt_vq_model.param_groups[0]['lr']
            }
            
            self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it, epoch_metrics)

            epoch += 1
            # if epoch % self.opt.save_every_e == 0:
            #     self.save(pjoin(self.opt.model_dir, 'E%04d.tar' % (epoch)), epoch, total_it=it)

            print('Validation time:')
            self.vq_model.eval()
            val_loss_rec = []
            val_loss_vel = []
            val_loss_commit = []
            val_loss = []
            val_perpexity = []
            val_loss_lpips = []
            with torch.no_grad():
                for i, batch_data in tqdm(enumerate(val_loader), desc = f"[Validation RVQ epoch {epoch}/{self.opt.max_epoch}]", total=len(val_loader)):
                    loss, loss_rec, loss_vel, loss_commit, perplexity, loss_lpips = self.forward(batch_data)
                    # val_loss_rec += self.l1_criterion(self.recon_motions, self.motions).item()
                    # val_loss_emb += self.embedding_loss.item()
                    val_loss.append(loss.item())
                    val_loss_rec.append(loss_rec.item())
                    val_loss_vel.append(loss_vel.item())
                    val_loss_commit.append(loss_commit.item())
                    val_perpexity.append(perplexity.item())
                    val_loss_lpips.append(loss_lpips.item() if isinstance(loss_lpips, torch.Tensor) else loss_lpips)

            # Calculate validation metrics
            val_metrics = {
                'loss': sum(val_loss) / len(val_loss),
                'loss_rec': sum(val_loss_rec) / len(val_loss_rec),
                'loss_explicit': sum(val_loss_vel) / len(val_loss_vel),
                'loss_commit': sum(val_loss_commit) / len(val_loss_commit),
                'perplexity': sum(val_perpexity) / len(val_perpexity),
                'loss_lpips': sum(val_loss_lpips) / len(val_loss_lpips) if val_loss_lpips else 0.0
            }
            
            # Log validation metrics
            self.logger.log_validation_metrics(val_metrics, epoch)

            # Print validation metrics including LPIPS if enabled
            lpips_str = f", LPIPS: {sum(val_loss_lpips)/len(val_loss_lpips):.5f}" if val_loss_lpips and any(val_loss_lpips) else ""
            print(f'Validation Loss: {sum(val_loss)/len(val_loss):.5f} Reconstruction: {sum(val_loss_rec)/len(val_loss):.5f}, '
                  f'Velocity: {sum(val_loss_vel)/len(val_loss):.5f}, Commit: {sum(val_loss_commit)/len(val_loss):.5f}{lpips_str}')

            # if sum(val_loss) / len(val_loss) < min_val_loss:
            #     min_val_loss = sum(val_loss) / len(val_loss)
            # # if sum(val_loss_vel) / len(val_loss_vel) < min_val_loss:
            # #     min_val_loss = sum(val_loss_vel) / len(val_loss_vel)
            #     min_val_epoch = epoch
            #     self.save(pjoin(self.opt.model_dir, 'finest.tar'), epoch, it)
            #     print('Best Validation Model So Far!~')
            # if self.opt.eval_on:
            if self.opt.dataset_name == 'cam':
                # Use camera-specific evaluation (now includes FID and other motion metrics)
                best_recon, best_smoothness, best_position_error, best_orientation_error, writer = evaluation_camera_vqvae(
                    self.opt.model_dir, eval_val_loader, self.vq_model, self.logger, epoch, 
                    best_recon=best_recon, best_smoothness=best_smoothness,
                    best_position_error=best_position_error, best_orientation_error=best_orientation_error,
                    eval_wrapper=eval_wrapper)
            else:
                # Use original evaluation for human motion
                best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer = evaluation_vqvae(
                    self.opt.model_dir, eval_val_loader, self.vq_model, self.logger, epoch, best_fid=best_fid,
                    best_div=best_div, best_top1=best_top1,
                    best_top2=best_top2, best_top3=best_top3, best_matching=best_matching, eval_wrapper=eval_wrapper)


            # if self.opt.eval_on and epoch % self.opt.eval_every_e == 0:
            data = torch.cat([self.motions[:4], self.pred_motion[:4]], dim=0).detach().cpu().numpy()
            # np.save(pjoin(self.opt.eval_dir, 'E%04d.npy' % (epoch)), data)
            save_dir = pjoin(self.opt.eval_dir, 'E%04d' % (epoch))
            os.makedirs(save_dir, exist_ok=True)
            if plot_eval is not None:
                plot_eval(data, save_dir)
                # if plot_eval is not None:
                #     save_dir = pjoin(self.opt.eval_dir, 'E%04d' % (epoch))
                #     os.makedirs(save_dir, exist_ok=True)
                #     plot_eval(data, save_dir)

            # if epoch - min_val_epoch >= self.opt.early_stop_e:
            #     print('Early Stopping!~')


class LengthEstTrainer(object):

    def __init__(self, args, estimator, text_encoder, encode_fnc):
        self.opt = args
        self.estimator = estimator
        self.text_encoder = text_encoder
        self.encode_fnc = encode_fnc
        self.device = args.device

        if args.is_train:
            # self.motion_dis
            try:
                from torch.utils.tensorboard import SummaryWriter  # type: ignore
            except ImportError:
                from torch.utils.tensorboard.writer import SummaryWriter  # type: ignore
            self.logger = SummaryWriter(args.log_dir)
            self.mul_cls_criterion = nn.CrossEntropyLoss()

    def resume(self, model_dir):
        checkpoints = torch.load(model_dir, map_location=self.device)
        self.estimator.load_state_dict(checkpoints['estimator'])
        # self.opt_estimator.load_state_dict(checkpoints['opt_estimator'])
        return checkpoints['epoch'], checkpoints['iter']

    def save(self, model_dir, epoch, niter):
        state = {
            'estimator': self.estimator.state_dict(),
            # 'opt_estimator': self.opt_estimator.state_dict(),
            'epoch': epoch,
            'niter': niter,
        }
        torch.save(state, model_dir)

    @staticmethod
    def zero_grad(opt_list):
        for opt in opt_list:
            opt.zero_grad()

    @staticmethod
    def clip_norm(network_list):
        for network in network_list:
            clip_grad_norm_(network.parameters(), 0.5)

    @staticmethod
    def step(opt_list):
        for opt in opt_list:
            opt.step()

    def train(self, train_dataloader, val_dataloader):
        self.estimator.to(self.device)
        self.text_encoder.to(self.device)

        self.opt_estimator = optim.Adam(self.estimator.parameters(), lr=self.opt.lr)

        epoch = 0
        it = 0

        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')
            epoch, it = self.resume(model_dir)

        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_dataloader)
        print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_dataloader), len(val_dataloader)))
        val_loss = 0
        min_val_loss = np.inf
        logs = defaultdict(float)
        while epoch < self.opt.max_epoch:
            # time0 = time.time()
            for i, batch_data in enumerate(train_dataloader):
                self.estimator.train()

                conds, _, m_lens = batch_data
                # word_emb = word_emb.detach().to(self.device).float()
                # pos_ohot = pos_ohot.detach().to(self.device).float()
                # m_lens = m_lens.to(self.device).long()
                text_embs = self.encode_fnc(self.text_encoder, conds, self.opt.device).detach()
                # print(text_embs.shape, text_embs.device)

                pred_dis = self.estimator(text_embs)

                self.zero_grad([self.opt_estimator])

                gt_labels = m_lens // self.opt.unit_length
                gt_labels = gt_labels.long().to(self.device)
                # print(gt_labels.shape, pred_dis.shape)
                # print(gt_labels.max(), gt_labels.min())
                # print(pred_dis)
                acc = (gt_labels == pred_dis.argmax(dim=-1)).sum() / len(gt_labels)
                loss = self.mul_cls_criterion(pred_dis, gt_labels)

                loss.backward()

                self.clip_norm([self.estimator])
                self.step([self.opt_estimator])

                logs['loss'] += loss.item()
                logs['acc'] += acc.item()

                it += 1
                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict({'val_loss': val_loss})
                    # self.logger.add_scalar('Val/loss', val_loss, it)

                    for tag, value in logs.items():
                        self.logger.add_scalar("Train/%s"%tag, value / self.opt.log_every, it)
                        mean_loss[tag] = value / self.opt.log_every
                    logs = defaultdict(float)
                    print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i)

                    if it % self.opt.save_latest == 0:
                        self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            epoch += 1

            print('Validation time:')

            val_loss = 0
            val_acc = 0
            # self.estimator.eval()
            with torch.no_grad():
                for i, batch_data in enumerate(val_dataloader):
                    self.estimator.eval()

                    conds, _, m_lens = batch_data
                    # word_emb = word_emb.detach().to(self.device).float()
                    # pos_ohot = pos_ohot.detach().to(self.device).float()
                    # m_lens = m_lens.to(self.device).long()
                    text_embs = self.encode_fnc(self.text_encoder, conds, self.opt.device)
                    pred_dis = self.estimator(text_embs)

                    gt_labels = m_lens // self.opt.unit_length
                    gt_labels = gt_labels.long().to(self.device)
                    loss = self.mul_cls_criterion(pred_dis, gt_labels)
                    acc = (gt_labels == pred_dis.argmax(dim=-1)).sum() / len(gt_labels)

                    val_loss += loss.item()
                    val_acc += acc.item()


            val_loss = val_loss / len(val_dataloader)
            val_acc = val_acc / len(val_dataloader)
            print('Validation Loss: %.5f Validation Acc: %.5f' % (val_loss, val_acc))

            if val_loss < min_val_loss:
                self.save(pjoin(self.opt.model_dir, 'finest.tar'), epoch, it)
                min_val_loss = val_loss

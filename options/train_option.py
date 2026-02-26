from options.base_option import BaseOptions
import argparse

class TrainT2MOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
        self.parser.add_argument('--max_epoch', type=int, default=500, help='Maximum number of epoch for training')
        # self.parser.add_argument('--max_iters', type=int, default=150_000, help='Training iterations')

        '''LR scheduler'''
        self.parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
        self.parser.add_argument('--gamma', type=float, default=0.1, help='Learning rate schedule factor')
        self.parser.add_argument('--milestones', default=[50_000], nargs="+", type=int,
                            help="learning rate schedule (iterations)")
        self.parser.add_argument('--warm_up_iter', default=2000, type=int, help='number of total iterations for warmup')

        '''Condition'''
        self.parser.add_argument('--cond_drop_prob', type=float, default=0.1, help='Drop ratio of condition, for classifier-free guidance')
        self.parser.add_argument("--seed", default=3407, type=int, help="Seed")
        self.parser.add_argument('--conditioning_mode', type=str, default='clip',
                                choices=['clip', 't5', 'id_embedding'],
                                help='Conditioning encoder: clip (default), t5 (token-level), id_embedding (per-sample learnable)')
        self.parser.add_argument('--num_id_samples', type=int, default=50,
                                help='Number of learnable sample embeddings for id_embedding mode')
        self.parser.add_argument('--t5_model_name', type=str, default='t5-base',
                                help='HuggingFace T5 model name for t5 conditioning mode')

        self.parser.add_argument('--is_continue', action="store_true", help='Is this trial continuing previous state?')
        self.parser.add_argument('--gumbel_sample', action="store_true", help='Strategy for token sampling, True: Gumbel sampling, False: Categorical sampling')
        self.parser.add_argument('--share_weight', action="store_true", help='Whether to share weight for projection/embedding, for residual transformer.')

        self.parser.add_argument('--log_every', type=int, default=50, help='Frequency of printing training progress, (iteration)')
        # self.parser.add_argument('--save_every_e', type=int, default=100, help='Frequency of printing training progress')
        self.parser.add_argument('--eval_every_e', type=int, default=5, help='Frequency of animating eval results, (epoch)')
        self.parser.add_argument('--save_latest', type=int, default=500, help='Frequency of saving checkpoint, (iteration)')

        '''Frame Conditioning'''
        self.parser.add_argument('--use_frames', action="store_true", help='Enable sparse keyframe conditioning with ResNet')
        self.parser.add_argument('--keyframe_arch', type=str, default='resnet18', choices=['resnet18', 'resnet34'], 
                                help='ResNet architecture for keyframe encoding')
        
        '''CLIP Fine-tuning'''
        self.parser.add_argument('--finetune_clip', action="store_true", help='Fine-tune CLIP last layers for camera direction understanding')
        self.parser.add_argument('--finetune_clip_layers', type=int, default=2, help='Number of CLIP transformer layers to unfreeze (default: 2)')
        self.parser.add_argument('--direction_loss_weight', type=float, default=0.1, help='Weight for direction contrastive loss (default: 0.1)')
        self.parser.add_argument('--smooth_loss_weight', type=float, default=0.0, help='Weight for trajectory smoothness regularization (default: 0.0 = disabled)')
        
        '''Mixed Precision Training'''
        self.parser.add_argument('--use_amp', action="store_true", help='Enable automatic mixed precision (FP16) training for 2x speedup')
        
        '''Logging & Experiment Tracking'''
        self.parser.add_argument('--use_wandb', action="store_true", help='Enable Weights & Biases logging')
        self.parser.add_argument('--wandb_project', type=str, default=None, help='W&B project name (default: momask-{dataset_name})')
        self.parser.add_argument('--wandb_entity', type=str, default=None, help='W&B entity/username (optional)')

        """Dataset"""
        # self.parser.add_argument('--data_suffix', type=str, default='', help='Suffix for dataset folder, e.g. _cam for camera datasets')

        self.is_train = True


class TrainLenEstOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.parser.add_argument('--name', type=str, default="test", help='Name of this trial')
        self.parser.add_argument("--gpu_id", type=int, default=-1, help='GPU id')

        self.parser.add_argument('--dataset_name', type=str, default='t2m', help='Dataset Name')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')

        self.parser.add_argument('--batch_size', type=int, default=64, help='Batch size')

        self.parser.add_argument("--unit_length", type=int, default=4, help="Length of motion")
        self.parser.add_argument("--max_text_len", type=int, default=20, help="Length of motion")

        self.parser.add_argument('--max_epoch', type=int, default=300, help='Maximum number of training epochs')

        self.parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')

        self.parser.add_argument('--is_continue', action="store_true", help='Resume training from latest checkpoint')

        self.parser.add_argument('--log_every', type=int, default=50, help='Frequency of printing training progress (iterations)')
        self.parser.add_argument('--save_every_e', type=int, default=5, help='Frequency of saving model checkpoint (epochs)')
        self.parser.add_argument('--eval_every_e', type=int, default=3, help='Frequency of running evaluation (epochs)')
        self.parser.add_argument('--save_latest', type=int, default=500, help='Frequency of saving latest checkpoint (iterations)')

    def parse(self):
        self.opt = self.parser.parse_args()
        self.opt.is_train = True
        # args = vars(self.opt)
        return self.opt

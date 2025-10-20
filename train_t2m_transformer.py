import os
import torch
import numpy as np

from torch.utils.data import DataLoader
from os.path import join as pjoin

from models.mask_transformer.transformer import MaskTransformer
from models.mask_transformer.transformer_trainer import MaskTransformerTrainer
from models.vq.model import RVQVAE

from options.train_option import TrainT2MOptions

from utils.plot_script import plot_3d_motion, plot_3d_motion_camera
from utils.unified_plotting import create_plotting_function_for_transformer
from utils.motion_process import recover_from_ric
from utils.get_opt import get_opt
from utils.fixseed import fixseed
from utils.paramUtil import t2m_kinematic_chain, kit_kinematic_chain
from utils.dataset_config import get_unified_dataset_config

from data.t2m_dataset import Text2MotionDataset, collate_fn_text2motion_camera_train
from motion_loaders.dataset_motion_loader import get_dataset_motion_loader
from models.t2m_eval_wrapper import EvaluatorModelWrapper


def plot_t2m(data, save_dir, captions, m_lengths):
    data = train_dataset.inv_transform(data)
    
    # Create unified plotting function that handles different camera formats automatically
    plot_function = create_plotting_function_for_transformer(opt.dataset_name)
    
    # Call the unified plotting function with all necessary parameters
    plot_function(
        data, save_dir, captions, m_lengths,
        fps=fps, 
        radius=radius, 
        joints_num=opt.joints_num,
        kinematic_chain=kinematic_chain
    )

def load_vq_model():
    opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'opt.txt')
    vq_opt = get_opt(opt_path, opt.device)
    vq_model = RVQVAE(vq_opt,
                dim_pose,
                vq_opt.nb_code,
                vq_opt.code_dim,
                vq_opt.output_emb_width,
                vq_opt.down_t,
                vq_opt.stride_t,
                vq_opt.width,
                vq_opt.depth,
                vq_opt.dilation_growth_rate,
                vq_opt.vq_act,
                vq_opt.vq_norm)
    
    # Choose checkpoint file based on dataset type
    is_camera_dataset = any(name in opt.dataset_name.lower() for name in ["cam", "estate", "realestate"])
    if is_camera_dataset:
        # For camera datasets, try different checkpoint files in order of preference
        checkpoint_files = [
            'net_best_recon.tar',      # Best reconstruction loss
            'net_best_position.tar',   # Best position accuracy
            'net_best_smoothness.tar', # Best smoothness
            'latest.tar'               # Latest checkpoint
        ]
        
        checkpoint_loaded = False
        for checkpoint_file in checkpoint_files:
            checkpoint_path = pjoin(vq_opt.checkpoints_dir, vq_opt.dataset_name, vq_opt.name, 'model', checkpoint_file)
            if os.path.exists(checkpoint_path):
                ckpt = torch.load(checkpoint_path, map_location='cpu')
                model_key = 'vq_model' if 'vq_model' in ckpt else 'net'
                vq_model.load_state_dict(ckpt[model_key])
                print(f'Loading VQ Model {opt.vq_name} from {checkpoint_file}')
                checkpoint_loaded = True
                break
        
        if not checkpoint_loaded:
            raise FileNotFoundError(f"No checkpoint found in {pjoin(vq_opt.checkpoints_dir, vq_opt.dataset_name, vq_opt.name, 'model')}")
    else:
        # For human motion datasets, use the original logic
        ckpt = torch.load(pjoin(vq_opt.checkpoints_dir, vq_opt.dataset_name, vq_opt.name, 'model', 'net_best_fid.tar'),
                                map_location='cpu')
        model_key = 'vq_model' if 'vq_model' in ckpt else 'net'
        vq_model.load_state_dict(ckpt[model_key])
        print(f'Loading VQ Model {opt.vq_name}')
    
    return vq_model, vq_opt

if __name__ == '__main__':
    parser = TrainT2MOptions()
    opt = parser.parse()
    fixseed(opt.seed)

    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
    # torch.autograd.set_detect_anomaly(True)  # Commented out for compatibility

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    # opt.meta_dir = pjoin(opt.save_root, 'meta')
    opt.eval_dir = pjoin(opt.save_root, 'animation')
    opt.log_dir = pjoin('./log/t2m/', opt.dataset_name, opt.name)

    os.makedirs(opt.model_dir, exist_ok=True)
    # os.makedirs(opt.meta_dir, exist_ok=True)
    os.makedirs(opt.eval_dir, exist_ok=True)
    os.makedirs(opt.log_dir, exist_ok=True)

    # Get unified dataset configuration with automatic format detection
    dataset_config = get_unified_dataset_config(opt)
    
    # Extract configuration values
    dim_pose = dataset_config['dim_pose']
    fps = dataset_config['fps']
    radius = dataset_config['radius']
    kinematic_chain = dataset_config['kinematic_chain']
    dataset_opt_path = dataset_config['dataset_opt_path']
    
    print(f"Dataset: {opt.dataset_name}")
    print(f"Data root: {opt.data_root}")
    print(f"Detected format: {dataset_config.get('detected_format', 'N/A')}")
    print(f"Feature dimensions: {dim_pose}")
    
    # Validate that dataset exists
    if not os.path.exists(opt.data_root):
        raise FileNotFoundError(f"Dataset directory does not exist: {opt.data_root}")
    if not os.path.exists(opt.motion_dir):
        raise FileNotFoundError(f"Motion directory does not exist: {opt.motion_dir}")
    
    # Set motion length for human motion datasets
    if opt.dataset_name in ['t2m', 'kit']:
        opt.max_motion_len = 55

    opt.text_dir = pjoin(opt.data_root, 'texts')

    vq_model, vq_opt = load_vq_model()

    clip_version = 'ViT-B/32'

    opt.num_tokens = vq_opt.nb_code

    t2m_transformer = MaskTransformer(code_dim=vq_opt.code_dim,
                                      cond_mode='text',
                                      latent_dim=opt.latent_dim,
                                      ff_size=opt.ff_size,
                                      num_layers=opt.n_layers,
                                      num_heads=opt.n_heads,
                                      dropout=opt.dropout,
                                      clip_dim=512,
                                      cond_drop_prob=opt.cond_drop_prob,
                                      clip_version=clip_version,
                                      opt=opt)

    # if opt.fix_token_emb:
    #     t2m_transformer.load_and_freeze_token_emb(vq_model.quantizer.codebooks[0])

    all_params = 0
    pc_transformer = sum(param.numel() for param in t2m_transformer.parameters_wo_clip())

    # print(t2m_transformer)
    # print("Total parameters of t2m_transformer net: {:.2f}M".format(pc_transformer / 1000_000))
    all_params += pc_transformer

    print('Total parameters of all models: {:.2f}M'.format(all_params / 1000_000))

    mean = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'meta', 'mean.npy'))
    std = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'meta', 'std.npy'))

    train_split_file = pjoin(opt.data_root, 'train.txt')
    val_split_file = pjoin(opt.data_root, 'val.txt')

    train_dataset = Text2MotionDataset(opt, mean, std, train_split_file)
    val_dataset = Text2MotionDataset(opt, mean, std, val_split_file)

    print(f"train.txt path: {train_split_file}")
    print(f"Number of training samples: {len(train_dataset)}")

    # Use camera-specific collate function for camera datasets
    is_camera_dataset = any(name in opt.dataset_name.lower() for name in ["cam", "estate", "realestate"])
    
    if is_camera_dataset:
        train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, num_workers=4, shuffle=True, drop_last=True, collate_fn=collate_fn_text2motion_camera_train)
        val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, num_workers=4, shuffle=True, drop_last=True, collate_fn=collate_fn_text2motion_camera_train)
    else:
        train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, num_workers=4, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, num_workers=4, shuffle=True, drop_last=True)
    # print(f"DEBUG: batch_size: {opt.batch_size}")
    # print(f"DEBUG: train_loader: {len(train_loader)}")
    # print(f"DEBUG: val_loader: {len(val_loader)}")
    # exit()
    eval_val_loader, _ = get_dataset_motion_loader(dataset_opt_path, 32, 'val', device=opt.device)

    wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
    # Add eval_on attribute - set to False for camera datasets since we don't have pretrained evaluation models
    wrapper_opt.eval_on = False if is_camera_dataset else True
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    trainer = MaskTransformerTrainer(opt, t2m_transformer, vq_model)

    trainer.train(train_loader, val_loader, eval_val_loader, eval_wrapper=eval_wrapper, plot_eval=plot_t2m)
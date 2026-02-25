"""
Evaluate TKCAM (Text-to-Camera-Motion) using CLaTr metrics
Integrates VQ-VAE + Transformer + Residual models with CLaTr evaluators
"""

import os
from os.path import join as pjoin
import torch
import numpy as np

from models.mask_transformer.transformer import MaskTransformer, ResidualTransformer
from models.vq.model import RVQVAE
from options.eval_option import EvalT2MOptions
from utils.get_opt import get_opt
from utils.dataset_config import get_unified_dataset_config
from motion_loaders.dataset_motion_loader import get_dataset_motion_loader
from utils.clatr_evaluator import CLaTrEvaluator, evaluate_tkcam_with_clatr
from utils.fixseed import fixseed


def load_vq_model(vq_opt, dim_pose, device):
    """Load VQ-VAE model"""
    vq_model = RVQVAE(
        vq_opt,
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
        vq_opt.vq_norm
    )
    
    ckpt = torch.load(
        pjoin(vq_opt.checkpoints_dir, vq_opt.dataset_name, vq_opt.name, 'model', 'net_best_fid.tar'),
        map_location=device
    )
    
    model_key = 'vq_model' if 'vq_model' in ckpt else 'net'
    vq_model.load_state_dict(ckpt[model_key])
    print(f'Loading VQ Model {vq_opt.name} from epoch {ckpt.get("ep", "unknown")}')
    
    return vq_model


def load_trans_model(model_opt, which_model, clip_version, device):
    """Load base transformer model"""
    t2m_transformer = MaskTransformer(
        code_dim=model_opt.code_dim,
        cond_mode='text',
        latent_dim=model_opt.latent_dim,
        ff_size=model_opt.ff_size,
        num_layers=model_opt.n_layers,
        num_heads=model_opt.n_heads,
        dropout=model_opt.dropout,
        clip_dim=512,
        cond_drop_prob=model_opt.cond_drop_prob,
        clip_version=clip_version,
        use_frames=getattr(model_opt, 'use_frames', False),
        opt=model_opt
    )
    
    # Try to load checkpoint - handle both .tar extensions
    checkpoint_path = pjoin(model_opt.checkpoints_dir, model_opt.dataset_name, model_opt.name, 'model', which_model)
    if not os.path.exists(checkpoint_path):
        # Try without extension if it was added
        base_name = which_model.replace('.tar', '')
        checkpoint_path = pjoin(model_opt.checkpoints_dir, model_opt.dataset_name, model_opt.name, 'model', base_name + '.tar')
    
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    # Check if model was trained with sparse keyframe encoder
    sparse_keyframe_encoder = None
    if 'sparse_keyframe_encoder' in ckpt:
        print("  MaskTransformer was trained with SparseKeyframeEncoder - loading it...")
        from models.sparse_keyframe_encoder import SparseKeyframeEncoder
        
        # Get config from opt if available, otherwise use defaults
        resnet_arch = getattr(model_opt, 'keyframe_arch', 'resnet18')
        latent_dim = model_opt.latent_dim
        
        sparse_keyframe_encoder = SparseKeyframeEncoder(
            resnet_arch=resnet_arch,
            latent_dim=latent_dim,
            pretrained=False  # Will load from checkpoint
        ).to(device)
        
        sparse_keyframe_encoder.load_state_dict(ckpt['sparse_keyframe_encoder'])
        sparse_keyframe_encoder.eval()
        print(f"  ✅ SparseKeyframeEncoder loaded for MaskTransformer ({resnet_arch})")
    
    model_key = 't2m_transformer' if 't2m_transformer' in ckpt else 'trans'
    missing_keys, unexpected_keys = t2m_transformer.load_state_dict(ckpt[model_key], strict=False)
    
    # Log any key mismatches but don't fail - filter out modality_emb_frame which is handled by sparse_keyframe_encoder
    unexpected_keys_filtered = [k for k in unexpected_keys if k != 'modality_emb_frame']
    
    if len(unexpected_keys_filtered) > 0:
        print(f'  Warning: {len(unexpected_keys_filtered)} unexpected keys in checkpoint')
        if len(unexpected_keys_filtered) <= 10:
            print(f'  Unexpected keys: {unexpected_keys_filtered}')
    
    if len(missing_keys) > 0:
        # Filter out clip_model keys which are expected to be missing
        missing_keys_filtered = [k for k in missing_keys if not k.startswith('clip_model.')]
        if len(missing_keys_filtered) > 0:
            print(f'  Warning: {len(missing_keys_filtered)} missing keys (non-clip)')
            if len(missing_keys_filtered) <= 10:
                print(f'  Missing keys: {missing_keys_filtered}')
    
    print(f'Loading Transformer {model_opt.name} from {which_model} (epoch {ckpt.get("ep", "unknown")})')
    
    return t2m_transformer, sparse_keyframe_encoder


def load_res_model(res_opt, vq_opt, clip_version, device):
    """Load residual transformer model"""
    res_opt.num_quantizers = vq_opt.num_quantizers
    res_opt.num_tokens = vq_opt.nb_code
    
    res_transformer = ResidualTransformer(
        code_dim=vq_opt.code_dim,
        cond_mode='text',
        latent_dim=res_opt.latent_dim,
        ff_size=res_opt.ff_size,
        num_layers=res_opt.n_layers,
        num_heads=res_opt.n_heads,
        dropout=res_opt.dropout,
        clip_dim=512,
        shared_codebook=vq_opt.shared_codebook,
        cond_drop_prob=res_opt.cond_drop_prob,
        share_weight=res_opt.share_weight,
        clip_version=clip_version,
        opt=res_opt
    )
    
    # Try to load best accuracy checkpoint first, fallback to best loss
    model_dir = pjoin(res_opt.checkpoints_dir, res_opt.dataset_name, res_opt.name, 'model')
    checkpoint_files = ['net_best_acc.tar', 'net_best_loss.tar', 'latest.tar']
    
    checkpoint_path = None
    for ckpt_file in checkpoint_files:
        test_path = pjoin(model_dir, ckpt_file)
        if os.path.exists(test_path):
            checkpoint_path = test_path
            print(f'Loading Residual Transformer from {ckpt_file}')
            break
    
    if checkpoint_path is None:
        raise FileNotFoundError(f"No checkpoint found in {model_dir}")
    
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    # Check if model was trained with sparse keyframe encoder
    sparse_keyframe_encoder = None
    if 'sparse_keyframe_encoder' in ckpt:
        print("  Model was trained with SparseKeyframeEncoder - loading it...")
        from models.sparse_keyframe_encoder import SparseKeyframeEncoder
        
        # Get config from opt if available, otherwise use defaults
        resnet_arch = getattr(res_opt, 'keyframe_arch', 'resnet18')
        latent_dim = res_opt.latent_dim
        
        sparse_keyframe_encoder = SparseKeyframeEncoder(
            resnet_arch=resnet_arch,
            latent_dim=latent_dim,
            pretrained=False  # Will load from checkpoint
        ).to(device)
        
        sparse_keyframe_encoder.load_state_dict(ckpt['sparse_keyframe_encoder'])
        sparse_keyframe_encoder.eval()
        print(f"  ✅ SparseKeyframeEncoder loaded ({resnet_arch})")
    
    missing_keys, unexpected_keys = res_transformer.load_state_dict(ckpt['res_transformer'], strict=False)
    
    # Log any key mismatches but don't fail - this is common with clip_model weights
    # Filter out the modality_emb_frame which is now handled by sparse_keyframe_encoder
    unexpected_keys_filtered = [k for k in unexpected_keys if k != 'modality_emb_frame']
    
    if len(unexpected_keys_filtered) > 0:
        print(f'  Warning: {len(unexpected_keys_filtered)} unexpected keys in checkpoint')
        if len(unexpected_keys_filtered) <= 10:
            print(f'  Unexpected keys: {unexpected_keys_filtered}')
    
    if len(missing_keys) > 0:
        # Filter out clip_model keys which are expected to be missing
        missing_keys_filtered = [k for k in missing_keys if not k.startswith('clip_model.')]
        if len(missing_keys_filtered) > 0:
            print(f'  Warning: {len(missing_keys_filtered)} missing keys (non-clip)')
            if len(missing_keys_filtered) <= 10:
                print(f'  Missing keys: {missing_keys_filtered}')
    
    print(f'Loading Residual Transformer {res_opt.name} from epoch {ckpt["ep"]}')
    
    return res_transformer, sparse_keyframe_encoder


if __name__ == '__main__':
    parser = EvalT2MOptions()
    parser.initialize()
    
    # Add CLaTr checkpoint arguments
    parser.parser.add_argument('--clatr_text_ckpt', type=str, 
                       default='/home/haozhe/CamTraj/CLaTr/lightning_logs/version_27/checkpoints/last.ckpt',
                       help='Path to CLaTr text-trajectory checkpoint')
    parser.parser.add_argument('--clatr_frame_ckpt', type=str, default=None,
                       help='Path to CLaTr frame-trajectory checkpoint (optional)')
    parser.parser.add_argument('--num_eval_samples', type=int, default=None,
                       help='Number of samples to evaluate (None = all)')
    
    # Frame loading arguments for evaluation
    parser.parser.add_argument('--load_frames', action='store_true',
                       help='Load frames for frame-trajectory evaluation')
    parser.parser.add_argument('--frame_dir', type=str,
                       default='/data4/haozhe/CamTraj/data/processed_estate/train_frames',
                       help='Directory containing frame images')
    parser.parser.add_argument('--max_frames', type=int, default=8,
                       help='Maximum number of frames to load per scene')
    
    # Debug mode for faster iteration
    parser.parser.add_argument('--debug', action='store_true',
                       help='Debug mode: use small batch size and limited samples')
    parser.parser.add_argument('--debug_samples', type=int, default=100,
                       help='Number of samples to use in debug mode')
    
    opt = parser.parse()
    fixseed(opt.seed)
    
    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
    torch.autograd.set_detect_anomaly(True)
    
    # Get unified dataset configuration (handles camera datasets with auto-detection)
    dataset_config = get_unified_dataset_config(opt)
    dim_pose = dataset_config['dim_pose']
    
    print(f"Dataset: {opt.dataset_name}")
    print(f"Detected dim_pose: {dim_pose}")
    if 'detected_format' in dataset_config and dataset_config['detected_format']:
        print(f"Detected format: {dataset_config['detected_format']}")
    
    clip_version = 'ViT-B/32'
    
    # Setup output directory
    root_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    model_dir = pjoin(root_dir, 'model')
    out_dir = pjoin(root_dir, 'eval_clatr')
    os.makedirs(out_dir, exist_ok=True)
    
    out_path = pjoin(out_dir, f"clatr_metrics_{opt.ext}.log")
    f = open(out_path, 'w')
    
    # Load model options
    model_opt_path = pjoin(root_dir, 'opt.txt')
    model_opt = get_opt(model_opt_path, device=opt.device)
    
    # Load VQ model
    vq_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'opt.txt')
    vq_opt = get_opt(vq_opt_path, device=opt.device)
    vq_model = load_vq_model(vq_opt, dim_pose, opt.device)
    
    model_opt.num_tokens = vq_opt.nb_code
    model_opt.num_quantizers = vq_opt.num_quantizers
    model_opt.code_dim = vq_opt.code_dim
    
    # Load residual model if specified AND exists
    res_model = None
    res_sparse_keyframe_encoder = None
    if hasattr(opt, 'res_name') and opt.res_name:
        res_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.res_name, 'opt.txt')
        
        # Check if residual model exists
        if os.path.exists(res_opt_path):
            print(f"\n{'='*80}")
            print(f"Loading Residual Transformer: {opt.res_name}")
            print("="*80)
            res_opt = get_opt(res_opt_path, device=opt.device)
            res_model, res_sparse_keyframe_encoder = load_res_model(res_opt, vq_opt, clip_version, opt.device)
            assert res_opt.vq_name == model_opt.vq_name
            print(f"✅ Residual model loaded successfully")
        else:
            print(f"\n⚠️  Warning: Residual model '{opt.res_name}' not found at {res_opt_path}")
            print("    Continuing with base transformer only (no residual refinement)")
            res_model = None
            res_sparse_keyframe_encoder = None
    
    # Initialize CLaTr evaluator
    print("\n" + "="*80)
    print("Initializing CLaTr Evaluator")
    print("="*80)
    
    clatr_evaluator = CLaTrEvaluator(
        text_ckpt_path=opt.clatr_text_ckpt,
        frame_ckpt_path=opt.clatr_frame_ckpt,
        device=str(opt.device)
    )
    
    # Setup dataloader
    opt.nb_joints = 21 if opt.dataset_name == 'kit' else 22
    
    # Debug mode configuration
    if opt.debug:
        print(f"\n{'='*80}")
        print("🐛 DEBUG MODE ENABLED")
        print("="*80)
        print(f"  Using only {opt.debug_samples} samples")
        print(f"  Batch size: 16 (reduced from 32)")
        print("="*80)
        eval_batch_size = 16
        if opt.num_eval_samples is None:
            opt.num_eval_samples = opt.debug_samples
    else:
        eval_batch_size = 32
    
    # Pass frame loading configuration to dataset
    if opt.load_frames:
        # Transfer frame config to model_opt so it gets passed to dataset
        model_opt.load_frames = True
        model_opt.frame_dir = opt.frame_dir
        model_opt.max_frames = opt.max_frames
        print(f"\n{'='*80}")
        print("Frame Loading Configuration")
        print("="*80)
        print(f"  Frames will be loaded for evaluation")
        print(f"  Frame directory: {opt.frame_dir}")
        print(f"  Max frames per scene: {opt.max_frames}")
        print("="*80)
    
    eval_val_loader, _ = get_dataset_motion_loader(
        model_opt_path, 
        batch_size=eval_batch_size, 
        fname='test',
        device=opt.device,
        load_frames=opt.load_frames
    )
    
    # Evaluate each checkpoint
    for file in os.listdir(model_dir):
        if opt.which_epoch != "all" and opt.which_epoch not in file:
            continue
        
        print(f'\n{"="*80}')
        print(f'Evaluating checkpoint: {file}')
        print("="*80)
        
        # Load transformer model
        t2m_transformer, trans_sparse_keyframe_encoder = load_trans_model(model_opt, file, clip_version, opt.device)
        t2m_transformer.eval()
        vq_model.eval()
        if res_model is not None:
            res_model.eval()
        if trans_sparse_keyframe_encoder is not None:
            trans_sparse_keyframe_encoder.eval()
        if res_sparse_keyframe_encoder is not None:
            res_sparse_keyframe_encoder.eval()
        
        t2m_transformer.to(opt.device)
        vq_model.to(opt.device)
        if res_model is not None:
            res_model.to(opt.device)
        if trans_sparse_keyframe_encoder is not None:
            trans_sparse_keyframe_encoder.to(opt.device)
        if res_sparse_keyframe_encoder is not None:
            res_sparse_keyframe_encoder.to(opt.device)
        
        # Run CLaTr evaluation
        # Note: For evaluation, we use the transformer's frame encoder (trans_sparse_keyframe_encoder)
        # The residual transformer's frame encoder (res_sparse_keyframe_encoder) is used during residual refinement
        results = evaluate_tkcam_with_clatr(
            eval_val_loader,
            vq_model,
            t2m_transformer,
            res_model,
            clatr_evaluator,
            num_samples=opt.num_eval_samples,
            dataset_type=opt.dataset_name,
            sparse_keyframe_encoder=trans_sparse_keyframe_encoder  # Use transformer's frame encoder
        )
        
        # Log results
        msg_header = f"\n{'='*80}\n{file} CLaTr Evaluation Results\n{'='*80}\n"
        print(msg_header)
        print(msg_header, file=f, flush=True)
        
        gen_metrics = results['generated']
        gt_metrics = results['ground_truth']
        
        msg_gen = f"Generated Trajectories (N={results['num_samples']}):\n"
        msg_gen += f"  Text→Traj: R@1={gen_metrics['t2m/R01']:.2f}%, R@5={gen_metrics['t2m/R05']:.2f}%, MedR={gen_metrics['t2m/MedR']:.1f}\n"
        msg_gen += f"  Traj→Text: R@1={gen_metrics['m2t/R01']:.2f}%, R@5={gen_metrics['m2t/R05']:.2f}%, MedR={gen_metrics['m2t/MedR']:.1f}\n"
        
        msg_gt = f"Ground Truth Trajectories:\n"
        msg_gt += f"  Text→Traj: R@1={gt_metrics['t2m/R01']:.2f}%, R@5={gt_metrics['t2m/R05']:.2f}%, MedR={gt_metrics['t2m/MedR']:.1f}\n"
        msg_gt += f"  Traj→Text: R@1={gt_metrics['m2t/R01']:.2f}%, R@5={gt_metrics['m2t/R05']:.2f}%, MedR={gt_metrics['m2t/MedR']:.1f}\n"
        
        # Add advanced metrics
        if 'advanced' in results:
            adv = results['advanced']
            msg_adv = f"\nAdvanced Metrics:\n"
            if adv.get('CLaTr-FID') is not None:
                msg_adv += f"  CLaTr-FID:      {adv['CLaTr-FID']:.2f} (lower is better)\n"
            if adv.get('Coverage') is not None:
                msg_adv += f"  Coverage:       {adv['Coverage']:.2f}% (higher is better)\n"
            if adv.get('CLaTr-CLIP-GT') is not None:
                msg_adv += f"  CLaTr-CLIP-GT:  {adv['CLaTr-CLIP-GT']:.2f}% (text-traj alignment)\n"
            if adv.get('CLaTr-CLIP-Gen') is not None:
                msg_adv += f"  CLaTr-CLIP-Gen: {adv['CLaTr-CLIP-Gen']:.2f}%\n"
            if adv.get('Diversity') is not None:
                msg_adv += f"  Diversity:      {adv['Diversity']:.2f}% (higher is better)\n"
        else:
            msg_adv = ""
        
        print(msg_gen)
        print(msg_gt)
        if msg_adv:
            print(msg_adv)
        print(msg_gen, file=f, flush=True)
        print(msg_gt, file=f, flush=True)
        if msg_adv:
            print(msg_adv, file=f, flush=True)
        
        # Save detailed metrics
        metric_file = pjoin(out_dir, f"{file.replace('.tar', '')}_clatr_metrics.npy")
        np.save(metric_file, results, allow_pickle=True)
        print(f"\nDetailed metrics saved to: {metric_file}")
    
    f.close()
    print(f"\n{'='*80}")
    print(f"Evaluation complete! Results saved to: {out_path}")
    print("="*80)

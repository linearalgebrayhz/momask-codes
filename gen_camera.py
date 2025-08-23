import os
from os.path import join as pjoin

import torch
import torch.nn.functional as F

from models.mask_transformer.transformer import MaskTransformer, ResidualTransformer
from models.vq.model import RVQVAE, LengthEstimator

from options.eval_option import EvalT2MOptions
from utils.get_opt import get_opt

from utils.fixseed import fixseed
from utils.camera_process import recover_from_camera_data, denormalize_camera_data
from utils.plot_script import plot_3d_motion

import numpy as np
from torch.distributions.categorical import Categorical
clip_version = 'ViT-B/32'

def load_vq_model(vq_opt):
    vq_model = RVQVAE(vq_opt,
                vq_opt.dim_pose,
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
    if vq_opt.dataset_name == "cam":
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
                print(f'Loading VQ Model {vq_opt.name} from {checkpoint_file} Completed!')
                checkpoint_loaded = True
                break
        
        if not checkpoint_loaded:
            raise FileNotFoundError(f"No VQ checkpoint found in {pjoin(vq_opt.checkpoints_dir, vq_opt.dataset_name, vq_opt.name, 'model')}")
    else:
        # For human motion datasets, use the original logic
        ckpt = torch.load(pjoin(vq_opt.checkpoints_dir, vq_opt.dataset_name, vq_opt.name, 'model', 'net_best_fid.tar'),
                                map_location='cpu')
        model_key = 'vq_model' if 'vq_model' in ckpt else 'net'
        vq_model.load_state_dict(ckpt[model_key])
        print(f'Loading VQ Model {vq_opt.name} Completed!')
    
    return vq_model, vq_opt

def load_trans_model(model_opt, opt, which_model):
    t2m_transformer = MaskTransformer(code_dim=model_opt.code_dim,
                                      cond_mode='text',
                                      latent_dim=model_opt.latent_dim,
                                      ff_size=model_opt.ff_size,
                                      num_layers=model_opt.n_layers,
                                      num_heads=model_opt.n_heads,
                                      dropout=model_opt.dropout,
                                      clip_dim=512,
                                      cond_drop_prob=model_opt.cond_drop_prob,
                                      clip_version=clip_version,
                                      opt=model_opt)
    ckpt = torch.load(pjoin(model_opt.checkpoints_dir, model_opt.dataset_name, model_opt.name, 'model', which_model),
                      map_location='cpu')
    model_key = 't2m_transformer' if 't2m_transformer' in ckpt else 'trans'
    missing_keys, unexpected_keys = t2m_transformer.load_state_dict(ckpt[model_key], strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])
    print(f'Loading Transformer {opt.name} from epoch {ckpt["ep"]}!')
    return t2m_transformer

def load_res_model(res_opt, vq_opt, opt):
    res_opt.num_quantizers = vq_opt.num_quantizers
    res_opt.num_tokens = vq_opt.nb_code
    res_transformer = ResidualTransformer(code_dim=vq_opt.code_dim,
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
                                            opt=res_opt)

    # Choose checkpoint file based on dataset type
    if res_opt.dataset_name == "cam":
        # For camera datasets, try different checkpoint files in order of preference
        checkpoint_files = [
            'net_best_acc.tar',        # Best accuracy
            'net_best_loss.tar',       # Best loss
            'latest.tar'               # Latest checkpoint
        ]
        
        checkpoint_loaded = False
        for checkpoint_file in checkpoint_files:
            checkpoint_path = pjoin(res_opt.checkpoints_dir, res_opt.dataset_name, res_opt.name, 'model', checkpoint_file)
            if os.path.exists(checkpoint_path):
                ckpt = torch.load(checkpoint_path, map_location=opt.device)
                checkpoint_loaded = True
                break
        
        if not checkpoint_loaded:
            raise FileNotFoundError(f"No residual transformer checkpoint found in {pjoin(res_opt.checkpoints_dir, res_opt.dataset_name, res_opt.name, 'model')}")
    else:
        # For human motion datasets, use the original logic
        ckpt = torch.load(pjoin(res_opt.checkpoints_dir, res_opt.dataset_name, res_opt.name, 'model', 'net_best_fid.tar'),
                          map_location=opt.device)
    missing_keys, unexpected_keys = res_transformer.load_state_dict(ckpt['res_transformer'], strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])
    print(f'Loading Residual Transformer {res_opt.name} from epoch {ckpt["ep"]}!')
    return res_transformer

def load_len_estimator(opt):
    model = LengthEstimator(512, 50)
    
    # Try to load length estimator with smart checkpoint loading
    estimator_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, 'length_estimator', 'model')
    
    # Define checkpoint files to try in order of preference
    checkpoint_files = ['finest.tar', 'latest.tar']
    
    # For camera dataset, also try fallback to t2m length estimator
    if opt.dataset_name == 'cam':
        t2m_estimator_dir = pjoin(opt.checkpoints_dir, 't2m', 'length_estimator', 'model')
        checkpoint_files.extend([
            pjoin(t2m_estimator_dir, 'finest.tar'),
            pjoin(t2m_estimator_dir, 'latest.tar')
        ])
    
    ckpt = None
    loaded_file = None
    
    # Try each checkpoint file
    for i, filename in enumerate(checkpoint_files):
        if i < 2:  # First two are in dataset-specific directory
            filepath = pjoin(estimator_dir, filename)
        else:  # Fallback files are already full paths
            filepath = filename
            
        if os.path.exists(filepath):
            try:
                ckpt = torch.load(filepath, map_location=opt.device)
                loaded_file = filepath
                break
            except Exception as e:
                print(f'Failed to load {filepath}: {e}')
                continue
    
    if ckpt is None:
        raise FileNotFoundError(f'No valid length estimator checkpoint found. Tried: {checkpoint_files}')
    
    model.load_state_dict(ckpt['estimator'])
    epoch = ckpt.get('epoch', 'unknown')
    
    if loaded_file and 't2m' in loaded_file and opt.dataset_name == 'cam':
        print(f'Loading Length Estimator from t2m dataset (epoch {epoch}) as fallback for camera dataset!')
    else:
        print(f'Loading Length Estimator from epoch {epoch}!')
    
    return model

def plot_camera_trajectory(data, save_path, title="Camera Trajectory"):
    """
    Plot camera trajectory as a 3D path
    """
    positions, orientations = recover_from_camera_data(data)
    
    # Create a simple 3D plot
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot camera positions
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2, label='Camera Path')
    ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], c='g', s=100, label='Start')
    ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], c='r', s=100, label='End')
    
    # Plot camera orientations as arrows at key points
    step = max(1, len(positions) // 10)  # Show every 10th orientation
    for i in range(0, len(positions), step):
        pos = positions[i]
        ori = orientations[i]
        
        # Convert pitch, yaw to direction vector
        pitch, yaw = ori[0], ori[1]
        dx = np.cos(pitch) * np.sin(yaw)
        dy = -np.sin(pitch)
        dz = np.cos(pitch) * np.cos(yaw)
        
        # Draw arrow
        ax.quiver(pos[0], pos[1], pos[2], dx, dy, dz, length=0.5, color='orange', alpha=0.7)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    parser = EvalT2MOptions()
    opt = parser.parse()
    fixseed(opt.seed)

    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
    torch.autograd.set_detect_anomaly(True)

    # Camera-specific configuration
    dim_pose = 5  # [x, y, z, pitch, yaw]

    root_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    model_dir = pjoin(root_dir, 'model')
    result_dir = pjoin('./generation', opt.ext)
    joints_dir = pjoin(result_dir, 'joints')
    animation_dir = pjoin(result_dir, 'animations')
    os.makedirs(joints_dir, exist_ok=True)
    os.makedirs(animation_dir, exist_ok=True)

    model_opt_path = pjoin(root_dir, 'opt.txt')
    model_opt = get_opt(model_opt_path, device=opt.device)

    #######################
    ######Loading RVQ######
    #######################
    vq_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'opt.txt')
    vq_opt = get_opt(vq_opt_path, device=opt.device)
    vq_opt.dim_pose = dim_pose
    vq_model, vq_opt = load_vq_model(vq_opt)

    model_opt.num_tokens = vq_opt.nb_code
    model_opt.num_quantizers = vq_opt.num_quantizers
    model_opt.code_dim = vq_opt.code_dim

    #################################
    ######Loading R-Transformer######
    #################################
    res_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.res_name, 'opt.txt')
    res_opt = get_opt(res_opt_path, device=opt.device)
    res_model = load_res_model(res_opt, vq_opt, opt)

    assert res_opt.vq_name == model_opt.vq_name

    #################################
    ######Loading M-Transformer######
    #################################
    t2m_transformer = load_trans_model(model_opt, opt, 'latest.tar')

    ##################################
    #####Loading Length Predictor#####
    ##################################
    length_estimator = load_len_estimator(model_opt)

    t2m_transformer.eval()
    vq_model.eval()
    res_model.eval()
    length_estimator.eval()

    res_model.to(opt.device)
    t2m_transformer.to(opt.device)
    vq_model.to(opt.device)
    length_estimator.to(opt.device)

    ##### ---- Dataloader ---- #####
    opt.nb_joints = 1  # Camera has no joints

    mean = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'meta', 'mean.npy'))
    std = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'meta', 'std.npy'))
    def inv_transform(data):
        return data * std + mean

    prompt_list = []
    length_list = []

    est_length = False
    if opt.text_prompt != "":
        prompt_list.append(opt.text_prompt)
        if opt.motion_length == 0:
            est_length = True
        else:
            length_list.append(opt.motion_length)
    elif opt.text_path != "":
        with open(opt.text_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                infos = line.split('#')
                prompt_list.append(infos[0])
                if len(infos) == 1 or (not infos[1].isdigit()):
                    est_length = True
                    length_list = []
                else:
                    length_list.append(int(infos[-1]))
    else:
        raise ValueError("A text prompt, or a file a text prompts are required!!!")

    if est_length:
        print("Since no motion length are specified, we will use estimated motion lengthes!!")
        text_embedding = t2m_transformer.encode_text(prompt_list)
        pred_dis = length_estimator(text_embedding)
        probs = F.softmax(pred_dis, dim=-1)  # (b, ntoken)
        token_lens = Categorical(probs).sample()  # (b, seqlen)
    else:
        token_lens = torch.LongTensor(length_list) // 4
        token_lens = token_lens.to(opt.device).long()

    m_length = token_lens * 4
    captions = prompt_list

    sample = 0

    for r in range(opt.repeat_times):
        print("-->Repeat %d"%r)
        with torch.no_grad():
            mids = t2m_transformer.generate(captions, token_lens,
                                            timesteps=opt.time_steps,
                                            cond_scale=opt.cond_scale,
                                            temperature=opt.temperature,
                                            topk_filter_thres=opt.topkr,
                                            gsample=opt.gumbel_sample)
            mids = res_model.generate(mids, captions, token_lens, temperature=1, cond_scale=5)
            pred_motions = vq_model.forward_decoder(mids)

            pred_motions = pred_motions.detach().cpu().numpy()

            data = inv_transform(pred_motions)

        for k, (caption, joint_data) in enumerate(zip(captions, data)):
            print("---->Sample %d: %s %d"%(k, caption, m_length[k]))
            animation_path = pjoin(animation_dir, str(k))
            joint_path = pjoin(joints_dir, str(k))

            os.makedirs(animation_path, exist_ok=True)
            os.makedirs(joint_path, exist_ok=True)

            joint_data = joint_data[:m_length[k]]
            
            # Save raw camera data
            np.save(pjoin(joint_path, "sample%d_repeat%d_len%d.npy"%(k, r, m_length[k])), joint_data)
            
            # Create camera trajectory visualization
            plot_path = pjoin(animation_path, "sample%d_repeat%d_len%d_trajectory.png"%(k, r, m_length[k]))
            plot_camera_trajectory(joint_data, plot_path, title=caption)
            
            # Save camera data as text file for easy inspection
            positions, orientations = recover_from_camera_data(joint_data)
            camera_data = np.column_stack([positions, orientations])
            np.savetxt(pjoin(joint_path, "sample%d_repeat%d_len%d.txt"%(k, r, m_length[k])), 
                      camera_data, fmt='%.6f', 
                      header='x y z pitch yaw', comments='')

            print(f"Camera trajectory saved to {plot_path}")
            print(f"Raw data saved to {pjoin(joint_path, 'sample%d_repeat%d_len%d.npy'%(k, r, m_length[k]))}") 
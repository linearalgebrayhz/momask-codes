from os.path import join as pjoin
import torch
from torch.utils import data
import numpy as np
from tqdm import tqdm
from torch.utils.data._utils.collate import default_collate
import random
import codecs as cs


def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)

def collate_fn_camera(batch):
    """
    Custom collate function for MotionDataset with camera trajectories.
    Pads all trajectories to the maximum length in the batch.
    """
    # For MotionDataset, each item is just a motion tensor
    # Sort by length (descending) to get the maximum length first
    batch.sort(key=lambda x: x.shape[0], reverse=True)
    
    # Get the maximum length in this batch
    max_len = batch[0].shape[0]
    
    # Pad all motions to the same length
    padded_batch = []
    for motion in batch:
        # Pad motion to max_len
        if motion.shape[0] < max_len:
            # Create padding with zeros
            padding = np.zeros((max_len - motion.shape[0], motion.shape[1]))
            motion = np.concatenate([motion, padding], axis=0)
        
        padded_batch.append(motion)
    
    # Use default collate on the padded batch
    return default_collate(padded_batch)

def collate_fn_text2motion_camera(batch):
    """
    Custom collate function for Text2MotionDatasetEval with camera trajectories.
    Pads all motions to the maximum length in the batch.
    Expected format: (word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token)
    or with frames: (word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token, keyframes)
    """
    # Check if batch includes frames
    has_frames = len(batch[0]) == 8
    
    # Sort by length (descending) to get the maximum length first
    if has_frames:
        batch.sort(key=lambda x: x[5], reverse=True)  # m_length is at index 5
    else:
        batch.sort(key=lambda x: x[5], reverse=True)  # m_length is at index 5
    
    # Get the maximum length in this batch
    max_len = batch[0][5]  # motion length is at index 5
    
    # Pad all motions to the same length
    padded_batch = []
    for item in batch:
        if has_frames:
            word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token, keyframes = item
        else:
            word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token = item
            keyframes = None
        
        # Pad motion to max_len
        if motion.shape[0] < max_len:
            # Create padding with zeros
            padding = np.zeros((max_len - motion.shape[0], motion.shape[1]))
            motion = np.concatenate([motion, padding], axis=0)
        
        # Update the item with padded motion
        if has_frames:
            padded_item = (word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token, keyframes)
        else:
            padded_item = (word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token)
        padded_batch.append(padded_item)
    
    # Use default collate on the padded batch
    return default_collate(padded_batch)


def collate_fn_text2motion_camera_train(batch):
    """
    Custom collate function for Text2MotionDataset with camera trajectories (training).
    Pads all motions to the maximum length in the batch.
    Expected format: (caption, motion, m_length)
    """
    # Sort by length (descending) to get the maximum length first
    batch.sort(key=lambda x: x[2], reverse=True)  # m_length is at index 2 for Text2MotionDataset
    
    # Get the maximum length in this batch
    max_len = batch[0][2]  # motion length is at index 2
    
    # Pad all motions to the same length
    padded_batch = []
    for item in batch:
        caption, motion, m_length = item
        
        # Pad motion to max_len
        if motion.shape[0] < max_len:
            # Create padding with zeros
            padding = np.zeros((max_len - motion.shape[0], motion.shape[1]))
            motion = np.concatenate([motion, padding], axis=0)
        
        # Update the item with padded motion
        padded_item = (caption, motion, m_length)
        padded_batch.append(padded_item)
    
    # Use default collate on the padded batch
    return default_collate(padded_batch)


def collate_fn_text2motion_camera_train_frames(batch):
    """
    Custom collate function for Text2MotionDataset with camera trajectories and frames (training).
    Pads all motions to the maximum length in the batch and handles frame PATHS (not loaded tensors).
    Expected format: (caption, motion, m_length, frame_paths)
    where frame_paths is a list of Path objects
    """
    # Check if batch includes frames
    has_frames = len(batch[0]) == 4
    
    if not has_frames:
        # Fallback to original collate
        return collate_fn_text2motion_camera_train(batch)
    
    # Sort by length (descending) to get the maximum length first
    batch.sort(key=lambda x: x[2], reverse=True)  # m_length is at index 2
    
    # Get the maximum length in this batch
    max_len = batch[0][2]  # motion length is at index 2
    
    # Separate components
    captions = []
    motions = []
    m_lengths = []
    frame_paths_list = []
    
    for item in batch:
        caption, motion, m_length, frame_paths = item  # frame_paths is list of Path objects
        
        # Pad motion to max_len
        if motion.shape[0] < max_len:
            padding = np.zeros((max_len - motion.shape[0], motion.shape[1]))
            motion = np.concatenate([motion, padding], axis=0)
        
        captions.append(caption)
        motions.append(motion)
        m_lengths.append(m_length)
        frame_paths_list.append(frame_paths)  # Keep as list of paths
    
    # Convert to tensors
    motions = torch.from_numpy(np.stack(motions, axis=0))
    m_lengths = torch.tensor(m_lengths)
    
    return captions, motions, m_lengths, frame_paths_list  # frame_paths_list is List[List[Path]]


class MotionDataset(data.Dataset):
    def __init__(self, opt, mean, std, split_file):
        self.opt = opt
        joints_num = opt.joints_num

        self.data = []
        self.lengths = []
        id_list = []
        with open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                if motion.shape[0] < opt.window_size:
                    continue
                self.lengths.append(motion.shape[0] - opt.window_size)
                self.data.append(motion)
            except Exception as e:
                # Some motion may not exist in KIT dataset
                print(e)
                pass

        self.cumsum = np.cumsum([0] + self.lengths)
        if opt.dataset_name == "cam":
            if opt.is_train:
                std = std / opt.feat_bias  # Scale all features by feat_bias
                # Save updated mean and std to meta_dir
                np.save(pjoin(opt.meta_dir, 'mean.npy'), mean)
                np.save(pjoin(opt.meta_dir, 'std.npy'), std)
            assert mean.shape[-1] == 5, f"Expected 5 features for cam dataset, got {mean.shape[-1]}"
        elif opt.dataset_name == "realestate10k_6":
            if opt.is_train:
                # Save updated mean and std to meta_dir for realestate10k_6
                np.save(pjoin(opt.meta_dir, 'mean.npy'), mean)
                np.save(pjoin(opt.meta_dir, 'std.npy'), std)
            assert mean.shape[-1] == 6, f"Expected 6 features for realestate10k_6 dataset, got {mean.shape[-1]}"
        elif opt.dataset_name == "realestate10k_12":
            if opt.is_train:
                # Save updated mean and std to meta_dir for realestate10k_12
                np.save(pjoin(opt.meta_dir, 'mean.npy'), mean)
                np.save(pjoin(opt.meta_dir, 'std.npy'), std)
            assert mean.shape[-1] == 12, f"Expected 12 features for realestate10k_12 dataset, got {mean.shape[-1]}"
        else: 
            if opt.is_train:
                # root_rot_velocity (B, seq_len, 1)
                std[0:1] = std[0:1] / opt.feat_bias
                # root_linear_velocity (B, seq_len, 2)
                std[1:3] = std[1:3] / opt.feat_bias
                # root_y (B, seq_len, 1)
                std[3:4] = std[3:4] / opt.feat_bias
                # ric_data (B, seq_len, (joint_num - 1)*3)
                std[4: 4 + (joints_num - 1) * 3] = std[4: 4 + (joints_num - 1) * 3] / 1.0
                # rot_data (B, seq_len, (joint_num - 1)*6)
                std[4 + (joints_num - 1) * 3: 4 + (joints_num - 1) * 9] = std[4 + (joints_num - 1) * 3: 4 + (
                        joints_num - 1) * 9] / 1.0
                # local_velocity (B, seq_len, joint_num*3)
                std[4 + (joints_num - 1) * 9: 4 + (joints_num - 1) * 9 + joints_num * 3] = std[
                                                                                        4 + (joints_num - 1) * 9: 4 + (
                                                                                                joints_num - 1) * 9 + joints_num * 3] / 1.0
                # foot contact (B, seq_len, 4)
                std[4 + (joints_num - 1) * 9 + joints_num * 3:] = std[
                                                                4 + (
                                                                            joints_num - 1) * 9 + joints_num * 3:] / opt.feat_bias

                assert 4 + (joints_num - 1) * 9 + joints_num * 3 + 4 == mean.shape[-1]
                np.save(pjoin(opt.meta_dir, 'mean.npy'), mean)
                np.save(pjoin(opt.meta_dir, 'std.npy'), std)

        self.mean = mean
        self.std = std
        print("Total number of motions {}, snippets {}".format(len(self.data), self.cumsum[-1]))

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return self.cumsum[-1]

    def __getitem__(self, item):
        if item != 0:
            motion_id = np.searchsorted(self.cumsum, item) - 1
            idx = item - self.cumsum[motion_id] - 1
        else:
            motion_id = 0
            idx = 0
        motion = self.data[motion_id][idx:idx + self.opt.window_size]
        "Z Normalization"
        motion = (motion - self.mean) / self.std

        return motion


class Text2MotionDatasetEval(data.Dataset):
    def __init__(self, opt, mean, std, split_file, w_vectorizer, load_frames=False):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = 20 #?
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        min_motion_len = 40 if self.opt.dataset_name =='t2m' else 24
        
        # Frame loading configuration for evaluation
        self.load_frames = load_frames
        if load_frames:
            from pathlib import Path
            import json
            from PIL import Image
            import torchvision.transforms as transforms
            
            # Frame preprocessing transforms (matching CLaTr)
            self.frame_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711]
                ),
            ])
            
            # Load scene_id_mapping
            mapping_path = Path(opt.data_root) / 'scene_id_mapping.json'
            if mapping_path.exists():
                with open(mapping_path) as f:
                    self.scene_id_mapping = json.load(f)
                print(f"✅ Loaded scene ID mapping: {len(self.scene_id_mapping)} entries")
            else:
                print(f"⚠️  Warning: scene_id_mapping.json not found at {mapping_path}")
                self.scene_id_mapping = {}
            
            # Frame directory
            frame_dir_str = getattr(opt, 'frame_dir', 
                                    '/data4/haozhe/CamTraj/data/processed_estate/train_frames')
            self.frame_dir = Path(frame_dir_str)
            self.max_frames = getattr(opt, 'max_frames', 8)
            
            print(f"✅ Frame loading enabled:")
            print(f"   - Frame dir: {self.frame_dir}")
            print(f"   - Max frames: {self.max_frames}")
            print(f"   - Frame size: 224x224")

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        # id_list = id_list[:250]
        # print("[Testing] id list:")
        # print("Length of id_list:", len(id_list))
        # print(id_list[:10])
        
        new_name_list = []
        length_list = []
        if opt.dataset_name == "cam" or opt.dataset_name == "realestate10k_6" or opt.dataset_name == "realestate10k_12":
            for name in tqdm(id_list):
                try:
                    # import pdb; pdb.set_trace()
                    motion_path = pjoin(opt.motion_dir, name + '.npy')
                    text_path = pjoin(opt.text_dir, name + '.txt')
                    motion = np.load(motion_path)
                    if (len(motion)) < min_motion_len or (len(motion) >= 300):
                        continue
                    text_data = []
                    flag = False
                    with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                        for line in f.readlines():
                            text_dict = {}
                            line_split = line.strip().split('#')
                            caption = line_split[0]
                            tokens = line_split[1].split(' ')
                            f_tag = 0.0
                            to_tag = 0.0
                            text_dict['caption'] = caption
                            text_dict['tokens'] = tokens 
                            # print(f"loading text file:\ncaption - {caption},\ntokens - {tokens},\n{f_tag}, {to_tag}")
                            if f_tag == 0.0 and to_tag == 0.0:
                                flag = True
                                text_data.append(text_dict)
                    if flag:
                        data_dict[name] = {'motion': motion,
                                        'length': len(motion),
                                        'text': text_data}
                        new_name_list.append(name)
                        length_list.append(len(motion))
                except:
                    print("error loading text")
        else:
            for name in tqdm(id_list):
                try:
                    motion_path = pjoin(opt.motion_dir, name + '.npy')
                    text_path = pjoin(opt.text_dir, name + '.txt')
                    print(f"Attempting to load motion: {motion_path}")
                    print(f"Attempting to load text: {text_path}")
                    motion = np.load(motion_path)
                    print(f"Motion loaded, shape: {motion.shape}")
                    # motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                    if (len(motion)) < min_motion_len or (len(motion) >= 300):
                        continue
                    text_data = []
                    flag = False
                    with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                        for line in f.readlines():
                            text_dict = {}
                            line_split = line.strip().split('#')
                            caption = line_split[0]
                            tokens = line_split[1].split(' ')
                            f_tag = float(line_split[2])
                            to_tag = float(line_split[3])
                            f_tag = 0.0 if np.isnan(f_tag) else f_tag
                            to_tag = 0.0 if np.isnan(to_tag) else to_tag

                            text_dict['caption'] = caption
                            text_dict['tokens'] = tokens
                            print(f"loading text file: \n caption {caption},\n tokens {tokens},\n {f_tag}, {to_tag}")
                            if f_tag == 0.0 and to_tag == 0.0:
                                flag = True
                                text_data.append(text_dict)
                            else:
                                try:
                                    n_motion = motion[int(f_tag*20) : int(to_tag*20)]
                                    if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                        continue
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                    while new_name in data_dict:
                                        new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                    data_dict[new_name] = {'motion': n_motion,
                                                        'length': len(n_motion),
                                                        'text':[text_dict]}
                                    new_name_list.append(new_name)
                                    length_list.append(len(n_motion))
                                except:
                                    print(line_split)
                                    print(line_split[2], line_split[3], f_tag, to_tag, name)
                                    # break

                    if flag:
                        data_dict[name] = {'motion': motion,
                                        'length': len(motion),
                                        'text': text_data}
                        new_name_list.append(name)
                        length_list.append(len(motion))
                except:
                    print("error loading text")
                
                

        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean
    
    def load_and_process_frames(self, scene_id):
        """
        Load and preprocess frames for a scene (for evaluation).
        
        Args:
            scene_id: Sequential scene ID (e.g., '000000')
        
        Returns:
            frames: Tensor of shape (max_frames, 3, 224, 224)
        """
        from pathlib import Path
        from PIL import Image
        import torch
        
        # Map sequential ID to hash ID
        hash_id = self.scene_id_mapping.get(scene_id, scene_id)
        scene_frame_dir = self.frame_dir / hash_id
        
        if not scene_frame_dir.exists():
            # Return dummy black frames
            return torch.zeros((self.max_frames, 3, 224, 224))
        
        # Get all frame files (sorted by name)
        frame_files = sorted(scene_frame_dir.glob('frame_*.jpg'))
        
        if len(frame_files) == 0:
            return torch.zeros((self.max_frames, 3, 224, 224))
        
        # Uniform sampling to get max_frames
        if len(frame_files) > self.max_frames:
            indices = np.linspace(0, len(frame_files) - 1, self.max_frames, dtype=int)
            frame_files = [frame_files[i] for i in indices]
        
        # Load and transform frames
        frames = []
        for frame_file in frame_files:
            try:
                img = Image.open(frame_file).convert("RGB")
                frame_tensor = self.frame_transform(img)
                frames.append(frame_tensor)
            except Exception as e:
                print(f"Warning: failed to load {frame_file}: {e}")
                continue
        
        if len(frames) == 0:
            return torch.zeros((self.max_frames, 3, 224, 224))
        
        # Stack frames
        frames_tensor = torch.stack(frames)  # (num_frames, 3, 224, 224)
        
        # Pad to max_frames if needed
        if frames_tensor.shape[0] < self.max_frames:
            padding = torch.zeros((self.max_frames - frames_tensor.shape[0], 3, 224, 224))
            frames_tensor = torch.cat([frames_tensor, padding], dim=0)
        
        return frames_tensor  # (max_frames, 3, 224, 224)

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']

        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        if self.opt.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.opt.unit_length) * self.opt.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]

        "Z Normalization"
        # import pdb; pdb.set_trace()
        # print(f"Motion shape: {motion.shape}")
        # print(f"Mean shape: {self.mean.shape}")
        # print(f"Std shape: {self.std.shape}")
        motion = (motion - self.mean) / self.std


        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)
        
        # Load frames if enabled (for evaluation)
        if self.load_frames:
            scene_id = self.name_list[idx]
            keyframes = self.load_and_process_frames(scene_id)
            return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), keyframes
        
        # print(word_embeddings.shape, motion.shape)
        # print(tokens)
        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens)


class Text2MotionDataset(data.Dataset):
    def __init__(self, opt, mean, std, split_file, load_frames=False):
        self.opt = opt
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        min_motion_len = 40 if self.opt.dataset_name =='t2m' else 24
        
        # Frame loading configuration
        self.load_frames = load_frames
        if load_frames:
            import json
            from pathlib import Path
            
            # Use configurable frame directory
            frame_dir_str = getattr(opt, 'frame_dir', 
                                    '/data4/haozhe/CamTraj/data/processed_estate/train_frames')
            self.frame_dir = Path(frame_dir_str)
            
            scene_id_mapping_path = Path(opt.data_root) / 'scene_id_mapping.json'
            with open(scene_id_mapping_path) as f:
                self.scene_id_mapping = json.load(f)
            
            print(f"Frame loading enabled. Frame dir: {self.frame_dir}")
            print(f"Scene ID mapping: {len(self.scene_id_mapping)} scenes")

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        # id_list = id_list[:250]
        
        print(f"\n{'='*60}")
        print(f"Loading dataset: {opt.dataset_name}")
        print(f"Split file: {split_file}")
        print(f"Total scenes to load: {len(id_list)}")
        print(f"{'='*60}\n")

        new_name_list = []
        length_list = []
        if opt.dataset_name == "cam" or opt.dataset_name == "realestate10k_6" or opt.dataset_name == "realestate10k_12":
            # Camera dataset: text files have format "caption#tokens" (2 fields)
            for name in tqdm(id_list, desc="Loading motion & text data"):
                try:
                    motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                    if (len(motion)) < min_motion_len or (len(motion) >= 300):
                        continue
                    text_data = []
                    flag = False
                    with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                        for line in f.readlines():
                            # print(line)
                            text_dict = {}
                            line_split = line.strip().split('#')
                            caption = line_split[0]
                            tokens = line_split[1].split(' ')
                            f_tag = 0.0
                            to_tag = 0.0
                            text_dict['caption'] = caption
                            text_dict['tokens'] = tokens
                            if f_tag == 0.0 and to_tag == 0.0:
                                flag = True
                                text_data.append(text_dict)
                    if flag:
                        data_dict[name] = {'motion': motion,
                                           'length': len(motion),
                                           'text': text_data}
                        new_name_list.append(name)
                        length_list.append(len(motion))
                except Exception as e:
                    print(f"Error loading {name}: {e}")
                    pass
        else:
            # Human motion dataset: text files have format "caption#tokens#f_tag#to_tag" (4 fields)
            for name in tqdm(id_list):
                try:
                    motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                    if (len(motion)) < min_motion_len or (len(motion) >= 300):
                        continue
                    text_data = []
                    flag = False
                    with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                        for line in f.readlines():
                            text_dict = {}
                            line_split = line.strip().split('#')
                            # print(line)
                            caption = line_split[0]
                            tokens = line_split[1].split(' ')
                            f_tag = float(line_split[2])
                            to_tag = float(line_split[3])
                            f_tag = 0.0 if np.isnan(f_tag) else f_tag
                            to_tag = 0.0 if np.isnan(to_tag) else to_tag

                            text_dict['caption'] = caption
                            text_dict['tokens'] = tokens
                            if f_tag == 0.0 and to_tag == 0.0:
                                flag = True
                                text_data.append(text_dict)
                            else:
                                try:
                                    n_motion = motion[int(f_tag*20) : int(to_tag*20)]
                                    if (len(n_motion)) < min_motion_len or (len(n_motion) >= 300):
                                        continue
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                    while new_name in data_dict:
                                        new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                    data_dict[new_name] = {'motion': n_motion,
                                                           'length': len(n_motion),
                                                           'text':[text_dict]}
                                    new_name_list.append(new_name)
                                    length_list.append(len(n_motion))
                                except:
                                    print(line_split)
                                    print(line_split[2], line_split[3], f_tag, to_tag, name)
                                    # break

                    if flag:
                        data_dict[name] = {'motion': motion,
                                           'length': len(motion),
                                           'text': text_data}
                        new_name_list.append(name)
                        length_list.append(len(motion))
                except Exception as e:
                    print(f"Error loading {name}: {e}")
                    pass

        # name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
        name_list, length_list = new_name_list, length_list

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list

    def inv_transform(self, data):
        return data * self.std + self.mean
    
    def load_scene_frames(self, scene_id, num_frames=None):
        """
        Load frame paths for a scene.
        
        Args:
            scene_id: Sequential ID (e.g., '000000')
            num_frames: Number of frames to return
                - If None: return ALL frame paths (for sparse keyframe sampling)
                - If int: uniformly sample that many frame paths
        
        Returns:
            frame_paths: List of Path objects to frame files (NOT loaded images!)
        """
        # Map scene_id to hash_id
        if scene_id not in self.scene_id_mapping:
            return []
        
        hash_id = self.scene_id_mapping[scene_id]
        scene_frame_dir = self.frame_dir / hash_id
        
        if not scene_frame_dir.exists():
            return []
        
        # Get all frame files, sorted by name
        frame_files = sorted(scene_frame_dir.glob('frame_*.jpg'))
        
        if num_frames is not None and len(frame_files) > num_frames:
            # Uniformly sample num_frames indices
            indices = np.linspace(0, len(frame_files) - 1, num_frames, dtype=int)
            frame_files = [frame_files[i] for i in indices]
        
        # Return ALL frame paths for sparse keyframe sampling
        # SparseKeyframeEncoder will randomly sample N∈[0,4] indices and load those images
        return frame_files

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']

        if self.opt.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.opt.unit_length) * self.opt.unit_length
        
        # Random temporal crop
        motion_start_idx = random.randint(0, len(motion) - m_length)
        motion = motion[motion_start_idx:motion_start_idx+m_length]

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)
        
        # Load frames if enabled - OPTIMIZED: Only return paths, not loaded images
        if self.load_frames:
            scene_id = self.name_list[idx]
            frame_paths = self.load_scene_frames(scene_id, num_frames=None)
            
            # Return frame paths as list (will be loaded in SparseKeyframeEncoder after sampling)
            return caption, motion, m_length, frame_paths
        
        # print(word_embeddings.shape, motion.shape)
        # print(tokens)
        return caption, motion, m_length

    def reset_min_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)
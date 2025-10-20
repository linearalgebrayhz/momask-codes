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
    """
    # Sort by length (descending) to get the maximum length first
    batch.sort(key=lambda x: x[5], reverse=True)  # m_length is at index 5 for Text2MotionDatasetEval
    
    # Get the maximum length in this batch
    max_len = batch[0][5]  # motion length is at index 5
    
    # Pad all motions to the same length
    padded_batch = []
    for item in batch:
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token = item
        
        # Pad motion to max_len
        if motion.shape[0] < max_len:
            # Create padding with zeros
            padding = np.zeros((max_len - motion.shape[0], motion.shape[1]))
            motion = np.concatenate([motion, padding], axis=0)
        
        # Update the item with padded motion
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
    def __init__(self, opt, mean, std, split_file, w_vectorizer):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = 20 #?
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        min_motion_len = 40 if self.opt.dataset_name =='t2m' else 24

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
        # print(word_embeddings.shape, motion.shape)
        # print(tokens)
        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens)


class Text2MotionDataset(data.Dataset):
    def __init__(self, opt, mean, std, split_file):
        self.opt = opt
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        min_motion_len = 40 if self.opt.dataset_name =='t2m' else 24

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        # id_list = id_list[:250]

        new_name_list = []
        length_list = []
        if opt.dataset_name == "cam" or opt.dataset_name == "realestate10k_6" or opt.dataset_name == "realestate10k_12":
            # Camera dataset: text files have format "caption#tokens" (2 fields)
            for name in tqdm(id_list):
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
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)
        # print(word_embeddings.shape, motion.shape)
        # print(tokens)
        return caption, motion, m_length

    def reset_min_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)
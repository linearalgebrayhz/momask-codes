from data.t2m_dataset import Text2MotionDatasetEval, collate_fn, collate_fn_text2motion_camera, collate_fn_text2motion_camera_train # TODO
from utils.word_vectorizer import WordVectorizer
import numpy as np
from os.path import join as pjoin
from torch.utils.data import DataLoader
from utils.get_opt import get_opt

def get_dataset_motion_loader(opt_path, batch_size, fname, device, load_frames=False):
    opt = get_opt(opt_path, device)

    # Configurations of T2M dataset and KIT dataset is almost the same
    if opt.dataset_name == 't2m' or opt.dataset_name == 'kit' or opt.dataset_name == 'cam':
        print('Loading dataset %s ...' % opt.dataset_name)

        mean = np.load(pjoin(opt.meta_dir, 'mean.npy'), allow_pickle=True)
        std = np.load(pjoin(opt.meta_dir, 'std.npy'), allow_pickle=True)

        w_vectorizer = WordVectorizer('./glove', 'our_vab')
        split_file = pjoin(opt.data_root, '%s.txt'%fname)
        dataset = Text2MotionDatasetEval(opt, mean, std, split_file, w_vectorizer, load_frames=load_frames)
        # Use camera-specific collate function for camera datasets
        if opt.dataset_name == 'cam':
            dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, drop_last=True,
                                    collate_fn=collate_fn_text2motion_camera, shuffle=True)
        else:
            dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, drop_last=True,
                                    collate_fn=collate_fn, shuffle=True)
    elif opt.dataset_name == 'realestate10k_6' or opt.dataset_name == 'realestate10k_12':
        print('Loading dataset %s ...' % opt.dataset_name)
        mean = np.load(pjoin(opt.meta_dir, 'mean.npy'), allow_pickle=True)
        std = np.load(pjoin(opt.meta_dir, 'std.npy'), allow_pickle=True)
        w_vectorizer = WordVectorizer('./glove', 'our_vab')
        split_file = pjoin(opt.data_root, '%s.txt'%fname)
        # print(f"opt.data_root: {opt.data_root}")
        # exit()
        dataset = Text2MotionDatasetEval(opt, mean, std, split_file, w_vectorizer, load_frames=load_frames)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, drop_last=True,collate_fn=collate_fn_text2motion_camera, shuffle=True) # collate fn for realestate10k_6 and realestate10k_6?
    else:
        raise KeyError('Dataset not Recognized !!')

    print('Ground Truth Dataset Loading Completed!!!')
    return dataloader, dataset
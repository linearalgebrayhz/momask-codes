import numpy as np
import os
import torch
from os.path import join as pjoin
from torch.utils.data import DataLoader

# Assuming these are part of the MoMask codebase
from data.t2m_dataset import Text2MotionDataset
from utils.word_vectorizer import WordVectorizer
from utils.get_opt import get_opt

def inspect_data():
    # Paths (adjust if different)
    dataset_root = "./dataset/KIT-ML/"
    glove_dir = "./glove"
    train_split_file = pjoin(dataset_root, "train.txt")
    dataset_opt_path = './checkpoints/t2m/Comp_v6_KLD005/opt.txt'  # Adjust to your config file

    # Load dataset options
    try:
        opt = get_opt(dataset_opt_path, torch.device("cpu"))
    except Exception as e:
        print(f"Error loading options: {e}")
        return

    opt.data_root = dataset_root
    opt.motion_dir = pjoin(dataset_root, "new_joint_vecs")
    opt.text_dir = pjoin(dataset_root, "texts")

    # Load mean and std
    mean = np.load(pjoin(dataset_root, "Mean.npy"))
    std = np.load(pjoin(dataset_root, "Std.npy"))
    print("Mean shape:", mean.shape)  # Expected: (251,)
    print("Std shape:", std.shape)   # Expected: (251,)
    print("Mean sample:", mean[:5])  # First 5 elements
    print("Std sample:", std[:5])

    # Load word vectorizer
    try:
        w_vectorizer = WordVectorizer(glove_dir, "our_vab")
    except Exception as e:
        print(f"Error loading word vectorizer: {e}")
        return

    # Initialize dataset
    try:
        train_dataset = Text2MotionDataset(opt, mean, std, train_split_file)
        print(f"Dataset size: {len(train_dataset)}")
    except Exception as e:
        print(f"Error creating dataset: {e}")
        return

    # Create data loader
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)

    # Inspect a batch
    for batch in train_loader:
        print("\nBatch structure:", type(batch), len(batch))
        print("Batch elements:", [type(x) for x in batch])

        # Correct unpacking
        if len(batch) >= 3:
            text_list, motion, m_length = batch[:3]
        else:
            print("Unexpected batch structure:", batch)
            return

        print("\nBatch details:")
        print("Motion type:", type(motion))
        if isinstance(motion, torch.Tensor):
            print("Motion shape:", motion.shape)  # Expected: (batch_size, 251, T)
            print("Motion sample (first sequence, first 5 features, first 5 frames):", motion[0, :5, :5])
        else:
            print("Motion content:", motion)

        print("Motion length type:", type(m_length))
        print("Motion length:", m_length)  # Expected: tensor([T1, T2, ...])
        print("Text list type:", type(text_list))
        print("Text list sample:", text_list)

        break  # Inspect one batch

    # Inspect a single sample
    try:
        sample = train_dataset[0]
        print("\nSingle sample structure:", type(sample), len(sample))
        print("Single sample elements:", [type(x) for x in sample])

        # Correct unpacking
        if len(sample) == 3:
            text_data, motion, m_length = sample
        else:
            print("Unexpected sample structure:", sample)
            return

        print("\nSingle sample details:")
        print("Motion type:", type(motion))
        if isinstance(motion, (torch.Tensor, np.ndarray)):
            print("Motion shape:", motion.shape)  # Expected: (T, 251)
            print("Motion sample:", motion[:, :])
        else:
            print("Motion content:", motion)

        print("Text data type:", type(text_data))
        print("Text data:", text_data)
        print("Motion length type:", type(m_length))
        print("Motion length:", m_length)
    except Exception as e:
        print(f"Error inspecting single sample: {e}")

if __name__ == "__main__":
    inspect_data()
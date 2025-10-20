from gen_camera import plot_camera_trajectory_animation
import numpy as np

from tqdm import tqdm

DATA_ROOT = "/home/haozhe/CamTraj/momask-codes/dataset/RealEstate10K_6feat_motion_test"

if __name__ == "__main__":
    ids = [f"{i:06d}" for i in range(0,20)]
    for id in tqdm(ids):
        try:
            data = np.load(f"{DATA_ROOT}/new_joint_vecs/{id}.npy")
            text = open(f"{DATA_ROOT}/untagged_text/{id}.txt", "r").read()
        except:
            continue
        plot_camera_trajectory_animation(data = data, 
                                        save_path = f"./dataset_vis/RealEstate10K_6feat_motion_test/{id}.mp4",
                                        title = text)
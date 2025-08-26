
# =====================================
# 4) dataset.py
# =====================================
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class BlocksDataset(Dataset):
    def __init__(self, root, split='train', points_per_block=4096, augment=True):
        super().__init__()
        self.files = sorted(glob.glob(os.path.join(root, split, '*.npz'))) or \
                     sorted(glob.glob(os.path.join(root, '*.npz')))  # fallback if not split dirs
        self.points_per_block = points_per_block
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        item = np.load(self.files[idx])
        feats = item['feats']  # [N, F]
        labels = item['labels']  # [N]

        N = feats.shape[0]
        if N >= self.points_per_block:
            choice = np.random.choice(N, self.points_per_block, replace=False)
        else:
            choice = np.random.choice(N, self.points_per_block, replace=True)
        pts = feats[choice]
        y = labels[choice]

        # Simple augmentations
        if self.augment:
            # random rotation around Z
            theta = np.random.uniform(0, 2*np.pi)
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float32)
            pts[:, :3] = pts[:, :3] @ R.T
            # jitter
            pts[:, :3] += np.random.normal(0, 0.01, size=pts[:, :3].shape).astype(np.float32)

        # Normalize XYZ to zero-mean per block
        xyz_mean = pts[:, :3].mean(0, keepdims=True)
        pts[:, :3] -= xyz_mean

        return torch.from_numpy(pts).float(), torch.from_numpy(y).long()


def make_loaders(root, batch_size=8, points_per_block=4096, num_workers=4):
    ds_train = BlocksDataset(os.path.join(root, 'train') if os.path.isdir(os.path.join(root, 'train')) else root,
                             split='train', points_per_block=points_per_block, augment=True)
    ds_val   = BlocksDataset(os.path.join(root, 'val') if os.path.isdir(os.path.join(root, 'val')) else root,
                             split='val', points_per_block=points_per_block, augment=False)
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    dl_val   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return dl_train, dl_val

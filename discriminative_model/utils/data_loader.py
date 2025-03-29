import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn.functional as F
from skimage import transform

from utils.utils import get_device


class MissingDataset(Dataset):
    def __init__(self, bscan_dir='./data/Training_Bscan'):
        self.bscan_dir = bscan_dir
        self.num_samples = 4400

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Load data
        file_path = os.path.join(self.bscan_dir, f'Bscan_{idx}.npy')
        bscan = np.load(file_path).astype(np.float32)
        b_min, b_max = bscan.min(), bscan.max()
        if b_max - b_min != 0:
            bscan_full = (bscan - b_min) / (b_max - b_min)
        else:
            bscan_full = np.zeros_like(bscan)

        # Simulate missing data:
        #   5% of the columns are randomly selected
        #   set a contiguous block (random length(1-6) columns) to zero（missing)
        bscan_missing = bscan_full.copy()
        missing_rate = 0.05
        n = bscan_full.shape[1]
        missing_start = np.random.choice(n, size=int(np.ceil(n*missing_rate)), replace=False)
        for s in missing_start:
            bscan_missing[:, s: min(s + np.random.randint(6) + 1, n)] = 0.0

        # Convert to torch.Tensor
        bscan_missing = torch.from_numpy(bscan_missing).unsqueeze(0)
        bscan_full    = torch.from_numpy(bscan_full).unsqueeze(0)
        return bscan_missing, bscan_full


class FWIDataset(Dataset):
    def __init__(self, bscan_dir='./data/Training_Bscan', labels_dir='./data/Training_Labels', num_samples=15000):
        self.bscan_dir = bscan_dir
        self.labels_dir = labels_dir
        self.num_samples = num_samples
        src = np.array([[0, 0],
                [230, 51],
                [230, 230],
                [0, 230]])

        dst = np.array([[0, 0],
                        [230, 0],
                        [230, 230],
                        [0, 230]])
        
        self.tform = transform.ProjectiveTransform()
        self.tform.estimate(dst, src)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        bscan_path = os.path.join(self.bscan_dir, f'Bscan_{idx}.npy')
        labels_path = os.path.join(self.labels_dir, f'Model_{idx}.npy')
        # Bscan data
        bscan = np.load(bscan_path, mmap_mode='r').astype(np.float32)
        b_min, b_max = bscan.min(), bscan.max()
        if b_max - b_min != 0:
            bscan = (bscan - b_min) / (b_max - b_min)
        else:
            bscan = np.zeros_like(bscan)
        # remove the useless area
        bscan = transform.warp(bscan, self.tform, output_shape=(230, 230))
        # Labels data
        labels_path = os.path.join(self.labels_dir, f'Model_{idx}.npy')
        labels = np.load(labels_path, mmap_mode='r').astype(np.float32)
        labels = (labels - 1.0) / 9.0 # Convert to [0.0, 1.0]

        # Convert to torch.Tensor
        bscan = torch.from_numpy(bscan).unsqueeze(0)
        labels = torch.from_numpy(labels).unsqueeze(0)
        return bscan, labels


def dataloader(dataset, batch_size, shuffle, train_ratio=0.8):
    # use GPU
    device, kwargs = get_device()

    total_len = len(dataset)
    train_len = int(total_len * train_ratio)
    test_len = total_len - train_len

    train_dataset = Subset(dataset, range(0, train_len))
    test_dataset  = Subset(dataset, range(train_len, total_len))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, **kwargs)

    return train_loader, test_loader


def dataloader_missing(batch_size=8, shuffle=True):
    dataset = MissingDataset()
    return dataloader(dataset, batch_size, shuffle)


def dataloader_fwi(batch_size=8, shuffle=True):
    dataset = FWIDataset()
    return dataloader(dataset, batch_size, shuffle)

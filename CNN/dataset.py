import os
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class JetImageDatasetRAM(Dataset):
    def __init__(self, h5_path):
        with h5py.File(h5_path, 'r') as f:
            # image: shape (N, 13, 32, 32)
            self.images = f['image'][:]
            # label: shape (N, 10)
            self.labels = f['label'][:]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return image, label


class JetImageDatasetStream(Dataset):
    def __init__(self, h5_path):
        self.h5_file = h5py.File(h5_path, 'r')
        self.images = self.h5_file['image']
        self.labels = self.h5_file['label']

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label

    def __del__(self):
        self.h5_file.close()


class JetImageDatasetWrapper(DataLoader):

    def __init__(self, h5_path, batch_size=64, num_workers=4, stream=False, shuffle=False):
        dataset_cls = JetImageDatasetStream if stream else JetImageDatasetRAM
        dataset = dataset_cls(h5_path)

        if stream:
            num_workers = 0

        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )

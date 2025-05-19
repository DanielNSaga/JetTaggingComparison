"""
HDF5 Dataset Loader for Jet Classification

This script provides utilities to load and process HDF5 datasets for jet classification tasks.
It supports two modes:
- **Streaming mode:** Loads data on demand from disk to save memory.
- **RAM mode:** Loads the entire dataset into memory for faster access.

The dataset is expected to contain:
- 'jet_label': One-hot encoded jet labels (shape: [N, 10])
- 'Pmu': 4-momentum of particles (shape: [N, n_nodes, 4])
- 'scalars': Additional scalar features for each node (shape: [N, n_nodes, d])
- 'atom_mask': Boolean mask for valid particles (shape: [N, n_nodes])
"""

import os
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy.sparse import coo_matrix
from torch.utils.data.distributed import DistributedSampler


def get_adj_matrix(n_nodes, batch_size, edge_mask):
    """
    Computes the adjacency matrix for particle interactions in the jet.

    Args:
        n_nodes (int): Number of nodes per jet.
        batch_size (int): Number of jets in a batch.
        edge_mask (torch.Tensor): Mask indicating valid particle interactions (shape: [B, n_nodes, n_nodes]).

    Returns:
        list: Edge index pairs for adjacency representation in graph-based models.
    """
    rows, cols = [], []
    for batch_idx in range(batch_size):
        base = batch_idx * n_nodes
        x = coo_matrix(edge_mask[batch_idx])
        rows.append(base + x.row)
        cols.append(base + x.col)
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    edges = [torch.LongTensor(rows), torch.LongTensor(cols)]
    return edges


def collate_fn(data):
    """
    Collate function for creating mini-batches.

    Args:
        data (list of tuples): Each element is (label, p4s, scalars, atom_mask).

    Returns:
        list: [labels, p4s, scalars, atom_mask, edge_mask, edges]
    """
    data = list(zip(*data))
    data = [torch.stack(item) for item in data]  # [labels, p4s, scalars, atom_mask]

    batch_size, n_nodes, _ = data[1].size()
    atom_mask = data[-1]  # [B, n_nodes]

    edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)  # [B, n_nodes, n_nodes]
    diag_mask = ~torch.eye(n_nodes, dtype=torch.bool).unsqueeze(0)
    edge_mask *= diag_mask

    edges = get_adj_matrix(n_nodes, batch_size, edge_mask)

    return data + [edge_mask, edges]


class H5Dataset(Dataset):
    """
    PyTorch Dataset class for loading HDF5 jet classification data.

    Supports two modes:
    - **Streaming mode**: Reads data on-the-fly from disk.
    - **RAM mode**: Loads the entire dataset into memory.

    Args:
        file_path (str): Path to the HDF5 file.
        load_to_ram (bool): If True, loads the entire dataset into memory.

    Attributes:
        data (dict or None): If load_to_ram=True, stores the dataset in memory.
        length (int): Number of samples in the dataset.
    """

    def __init__(self, file_path, load_to_ram=False):
        self.file_path = file_path
        self.load_to_ram = load_to_ram
        self.data = None

        with h5py.File(file_path, 'r') as f:
            self.length = f['jet_label'].shape[0]
            if load_to_ram:
                print(f"Loading dataset {file_path} into RAM...")
                self.data = {key: torch.tensor(f[key][:]) for key in f.keys()}
                print(f"Dataset {file_path} loaded successfully.")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.load_to_ram:
            label = torch.argmax(self.data['jet_label'][idx])  # Convert one-hot label to integer class
            p4s = self.data['Pmu'][idx]
            scalars = self.data['scalars'][idx]
            atom_mask = self.data['atom_mask'][idx].bool()
        else:
            with h5py.File(self.file_path, 'r') as f:
                label = torch.argmax(torch.tensor(f['jet_label'][idx]))
                p4s = torch.tensor(f['Pmu'][idx])
                scalars = torch.tensor(f['scalars'][idx])
                atom_mask = torch.tensor(f['atom_mask'][idx]).bool()

        return label, p4s, scalars, atom_mask


def retrieve_dataloaders(batch_size, data_dir="./Data", num_workers=4, load_to_ram=False, pin_memory=True):
    """
    Creates DataLoaders for training, validation, and testing from HDF5 files.

    Args:
        batch_size (int): Batch size for training and evaluation.
        data_dir (str): Directory where HDF5 files are stored.
        num_workers (int): Number of worker processes for data loading.
        load_to_ram (bool): If True, loads datasets into RAM for faster access.

    Returns:
        tuple: (train_sampler, dataloaders) where dataloaders is a dictionary with 'train', 'val', 'test'.
    """
    splits = ['train', 'val', 'test']
    datasets = {split: H5Dataset(os.path.join(data_dir, f"{split}.h5"), load_to_ram) for split in splits}

    # Check if distributed training is enabled
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            print("Distributed training detected. Using DistributedSampler.")
            samplers = {split: DistributedSampler(datasets[split], shuffle=(split == 'train')) for split in splits}
        else:
            samplers = {split: None for split in splits}
    except Exception:
        samplers = {split: None for split in splits}

    dataloaders = {
        split: DataLoader(
            datasets[split],
            batch_size=batch_size,
            sampler=samplers[split],
            num_workers=num_workers,
            shuffle=(split == 'train' and samplers[split] is None),
            pin_memory=True,
            drop_last=(split == 'train'),
            collate_fn=collate_fn,
            persistent_workers=True
        ) for split in splits
    }

    print("DataLoaders initialized successfully.")
    return samplers['train'], dataloaders


if __name__ == "__main__":
    # Example usage
    train_sampler, dataloaders = retrieve_dataloaders(batch_size=32, data_dir="./Data", num_workers=4, load_to_ram=True)
    print("Dataloaders ready for use.")

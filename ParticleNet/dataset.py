"""
dataset.py

This module defines a PyTorch-compatible dataset class for loading padded particle-level
data from HDF5 files created for jet tagging tasks (e.g., ParticleNet).

It supports:
- Static loading (entire file loaded into memory)
- Streaming mode (per-sample reading, lower memory usage)
- Flexible feature selection (particle coordinates, energies, etc.)
- Output formatting (channel_first or channel_last)

Expected HDF5 structure:
- Each dataset contains:
    - Feature arrays (e.g., part_pt, part_eta, part_phi, part_energy)
    - Labels: one-hot or integer labels under the key "label" (default)

Typical output per sample:
{
    "X": {
        "points": (pad_len, 3) or (3, pad_len),
        "features": (pad_len, C) or (C, pad_len)
    },
    "y": label
}
"""

import os
import glob
import logging
import numpy as np
import torch
import h5py
from torch.utils.data import Dataset as TorchDataset

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')


def pad_array(a, maxlen, value=0., dtype='float32'):
    """
    Pads each row of a list of arrays to a fixed length.

    Args:
        a (list of arrays): Array per event (e.g., particle list).
        maxlen (int): Maximum number of entries per event.
        value (float): Padding value.
        dtype (str): Output dtype.

    Returns:
        np.ndarray: Shape (num_events, maxlen)
    """
    x = (np.ones((len(a), maxlen)) * value).astype(dtype)
    for idx, s in enumerate(a):
        if not len(s):
            continue
        trunc = s[:maxlen].astype(dtype)
        x[idx, :len(trunc)] = trunc
    return x


class H5Dataset(TorchDataset):
    """
    PyTorch Dataset for loading jet tagging data from HDF5 files.

    Supports full in-memory or streaming access per event. Each sample consists
    of a dictionary with particle features and a label.

    Args:
        filepath (str): Path to HDF5 file.
        feature_dict (dict): Keys are feature groups (e.g. 'points', 'features').
                             Values are lists of HDF5 keys to include.
        label (str): Name of the label key in the file.
        pad_len (int): Number of particles per event after padding.
        data_format (str): 'channel_first' or 'channel_last'.
        stream (bool): Whether to stream data on __getitem__ or load into memory.
    """
    def __init__(self, filepath, feature_dict=None, label='label', pad_len=128,
                 data_format='channel_last', stream=False):
        self.filepath = filepath
        self.label = label
        self.pad_len = pad_len
        self._stream = stream
        self.stack_axis = 1 if data_format == 'channel_first' else -1

        if feature_dict is None:
            feature_dict = {
                'points': ['part_eta', 'part_phi', 'part_pt'],
                'features': ['part_pt', 'part_eta', 'part_phi', 'part_energy']
            }
        self.feature_dict = feature_dict

        if not self._stream:
            logging.info(f'Loading entire file into memory: {self.filepath}')
            with h5py.File(self.filepath, "r") as f:
                self._label = f[self.label][:]
                self._values = {}
                for key, cols in self.feature_dict.items():
                    if not isinstance(cols, (list, tuple)):
                        cols = [cols]
                    arrs = []
                    for col in cols:
                        data = f[col][:]
                        padded = pad_array(data, self.pad_len)
                        arrs.append(padded)
                    self._values[key] = np.stack(arrs, axis=self.stack_axis)
            logging.info(f'Finished loading {self.filepath}')
        else:
            with h5py.File(self.filepath, "r") as f:
                self._length = f[self.label].shape[0]
            self._label = None
            self._values = None

    def __len__(self):
        return self._length if self._stream else len(self._label)

    def __getitem__(self, index):
        if self._stream:
            with h5py.File(self.filepath, "r") as f:
                sample = {}
                sample_label = f[self.label][index]
                for key, cols in self.feature_dict.items():
                    if not isinstance(cols, (list, tuple)):
                        cols = [cols]
                    arrs = []
                    for col in cols:
                        raw = f[col][index]
                        padded = pad_array([raw], self.pad_len)[0]
                        arrs.append(padded)
                    sample[key] = np.stack(arrs, axis=self.stack_axis)
                return {"X": sample, "y": sample_label}
        else:
            sample = {key: self._values[key][index] for key in self._values}
            return {"X": sample, "y": self._label[index]}


def get_datasets(train_path, val_path, test_path, **kwargs):
    """
    Loads three datasets (train/val/test) from HDF5 files.

    Args:
        train_path (str): Path to train.h5
        val_path (str): Path to val.h5
        test_path (str): Path to test.h5
        **kwargs: Passed to each H5Dataset

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    train_ds = H5Dataset(train_path, **kwargs)
    val_ds   = H5Dataset(val_path, **kwargs)
    test_ds  = H5Dataset(test_path, **kwargs)
    return train_ds, val_ds, test_ds

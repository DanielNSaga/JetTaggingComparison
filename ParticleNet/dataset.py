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
        a (list of arrays]): Array per event (e.g., particle list).
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
        feature_dict (dict): Keys are 'points', 'features' (or any grouping name).
                             Values are lists of HDF5 keys to include.
        label (str): Name of the label dataset in the file.
        pad_len (int): Number of particles per event after padding.
        data_format (str): 'channel_first' or 'channel_last'.
        stream (bool): Whether to stream data on __getitem__ or load everything into memory.
    """
    def __init__(self, filepath, feature_dict=None, label='label', pad_len=128,
                 data_format='channel_last', stream=False):
        self.filepath = filepath
        self.label = label
        self.pad_len = pad_len
        self._stream = stream
        # Hvis 'channel_first', stackes kolonner på axis=1; ellers axis=-1
        self.stack_axis = 1 if data_format == 'channel_first' else -1

        # -- Juster feature_dict til de nye kolonne-navnene i HDF5 --
        if feature_dict is None:
            # Eksempel: "points" inneholder kun (delta_eta, delta_phi),
            # mens "features" inneholder resterende
            feature_dict = {
                "points": [
                    "part_delta_eta",
                    "part_delta_phi"
                ],
                "features": [
                    "part_log_pt",
                    "part_log_energy",
                    "part_log_ptrel",
                    "part_log_Erel",
                    "part_deltaR",
                    "part_charge",
                    "part_isElectron",
                    "part_isMuon",
                    "part_isChargedHadron",
                    "part_isNeutralHadron",
                    "part_isPhoton"
                ]
            }
        self.feature_dict = feature_dict

        if not self._stream:
            logging.info(f'Loading entire file into memory: {self.filepath}')
            with h5py.File(self.filepath, "r") as f:
                # Les inn label (eks. one-hot vektor)
                self._label = f[self.label][:]
                self._values = {}
                for key, cols in self.feature_dict.items():
                    if not isinstance(cols, (list, tuple)):
                        cols = [cols]
                    arrs = []
                    for col in cols:
                        data = f[col][:]
                        arrs.append(data)
                    self._values[key] = np.stack(arrs, axis=self.stack_axis)
            logging.info(f'Finished loading {self.filepath}')
            self._length = len(self._label)
        else:
            with h5py.File(self.filepath, "r") as f:
                self._length = f[self.label].shape[0]
            self._label = None
            self._values = None

    def __len__(self):
        return self._length

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
                        arrs.append(raw)
                    sample[key] = np.stack(arrs, axis=self.stack_axis)
                return {"X": sample, "y": sample_label}
        else:
            sample = {key: self._values[key][index] for key in self._values}
            sample_label = self._label[index]
            return {"X": sample, "y": sample_label}


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

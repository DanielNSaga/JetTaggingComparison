"""
convert_file.py

This script converts ROOT files containing jet data into structured HDF5 files
for use in machine learning pipelines such as ParticleNet. It performs the following:

1. Loads ROOT files and extracts relevant particle and jet features.
2. Computes derived per-particle features such as:
   - log(pt), log(energy), delta eta, delta phi, delta R
   - relative log(pt) and log(energy)
   - charge and particle type (electron, muon, etc.)
3. Pads jagged arrays to fixed-size particle tensors (N x MAX_PARTICLES).
4. Splits each file into train/val/test subsets with consistent size across files.
5. Saves the resulting data to compressed HDF5 files in `ParticleNet/Data`.

Author: [Your Name]
"""

import os
import glob
import logging
import numpy as np
import torch
import h5py
import uproot

# === Setup paths ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
SOURCE_DIR = "../Data"
DEST_DIR = os.path.join(SCRIPT_DIR, "Data")

os.makedirs(DEST_DIR, exist_ok=True)
ROOT_FILES = glob.glob(os.path.join(SOURCE_DIR, "*.root"))

# === Logging ===
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

# === Constants ===
MAX_PARTICLES = 128
LABEL_COLS = [
    'label_QCD', 'label_Hbb', 'label_Hcc', 'label_Hgg',
    'label_H4q', 'label_Hqql', 'label_Zqq', 'label_Wqq',
    'label_Tbqq', 'label_Tbl'
]

EPS = 1e-6


def pad_event(arr, max_len, pad_value=0.0):
    """
    Pads or truncates a 1D array to a fixed length.

    Args:
        arr (array-like): Input array.
        max_len (int): Desired output length.
        pad_value (float): Value to use for padding.

    Returns:
        np.ndarray: Array of shape (max_len,)
    """
    arr = np.asarray(arr, dtype=np.float32)
    if arr.shape[0] >= max_len:
        return arr[:max_len]
    pad = np.full((max_len - arr.shape[0],), pad_value, dtype=np.float32)
    return np.concatenate([arr, pad])


def transform_dataframe(df, max_particles=128, eps=1e-6):
    """
    Transforms a DataFrame into a dictionary of padded per-particle features,
    computing additional physics-aware quantities.

    Returns:
        dict[str, np.ndarray]: Dictionary containing features of shape (N, max_particles),
                               and label of shape (N, 10)
    """
    # Extract and pad 4-vector components
    px = np.stack([pad_event(p, max_particles) for p in df['part_px']])
    py = np.stack([pad_event(p, max_particles) for p in df['part_py']])
    pz = np.stack([pad_event(p, max_particles) for p in df['part_pz']])
    E  = np.stack([pad_event(p, max_particles) for p in df['part_energy']])

    # Precomputed deltas
    delta_eta = np.stack([pad_event(p, max_particles) for p in df['part_deta']])
    delta_phi = np.stack([pad_event(p, max_particles) for p in df['part_dphi']])

    # Particle type and charge flags
    part_charge          = np.stack([pad_event(p, max_particles) for p in df['part_charge']])
    part_isElectron      = np.stack([pad_event(p, max_particles) for p in df['part_isElectron']])
    part_isMuon          = np.stack([pad_event(p, max_particles) for p in df['part_isMuon']])
    part_isChargedHadron = np.stack([pad_event(p, max_particles) for p in df['part_isChargedHadron']])
    part_isNeutralHadron = np.stack([pad_event(p, max_particles) for p in df['part_isNeutralHadron']])
    part_isPhoton        = np.stack([pad_event(p, max_particles) for p in df['part_isPhoton']])

    # Convert to torch tensors for easier math
    px_t, py_t, pz_t, E_t = map(torch.tensor, (px, py, pz, E))
    px_t, py_t, pz_t, E_t = px_t.float(), py_t.float(), pz_t.float(), E_t.float()
    mask = (E_t > 0).float()

    pt_t = torch.sqrt(px_t**2 + py_t**2 + eps)
    sum_pt_t = (pt_t * mask).sum(dim=1, keepdim=True)
    sum_E_t  = (E_t * mask).sum(dim=1, keepdim=True)

    log_pt     = torch.log(pt_t + eps)
    log_energy = torch.log(E_t + eps)
    log_ptrel  = log_pt - torch.log(sum_pt_t + eps)
    log_Erel   = log_energy - torch.log(sum_E_t + eps)

    delta_eta_t = torch.tensor(delta_eta, dtype=torch.float32)
    delta_phi_t = torch.tensor(delta_phi, dtype=torch.float32)
    deltaR_t = torch.sqrt(delta_eta_t**2 + delta_phi_t**2 + eps)

    # One-hot labels
    labels = np.stack([df[col].values.astype(int) for col in LABEL_COLS], axis=1)

    return {
        "part_delta_eta": delta_eta_t.numpy(),
        "part_delta_phi": delta_phi_t.numpy(),
        "part_log_pt": log_pt.numpy(),
        "part_log_energy": log_energy.numpy(),
        "part_log_ptrel": log_ptrel.numpy(),
        "part_log_Erel": log_Erel.numpy(),
        "part_deltaR": deltaR_t.numpy(),
        "part_charge": part_charge,
        "part_isElectron": part_isElectron,
        "part_isMuon": part_isMuon,
        "part_isChargedHadron": part_isChargedHadron,
        "part_isNeutralHadron": part_isNeutralHadron,
        "part_isPhoton": part_isPhoton,
        "label": labels
    }


# === Pass 1: determine consistent split sizes across all files ===
train_counts, test_counts, val_counts = [], [], []

for file in ROOT_FILES:
    try:
        df = uproot.open(file)["tree"].arrays(library="pd")
        data = transform_dataframe(df, MAX_PARTICLES)
        n = data["label"].shape[0]
        n_train = int(n * 0.8)
        n_test = int(n * 0.1)
        n_val = n - n_train - n_test
        train_counts.append(n_train)
        test_counts.append(n_test)
        val_counts.append(n_val)
        logging.info(f"{os.path.basename(file)}: {n} events -> train {n_train}, test {n_test}, val {n_val}")
    except Exception as e:
        logging.error(f"Failed to process {file}: {e}")

if not train_counts:
    raise RuntimeError("No usable ROOT files found.")

common_train = min(train_counts)
common_test = min(test_counts)
common_val = min(val_counts)
logging.info(f"Common split per file: train={common_train}, test={common_test}, val={common_val}")


# === Prepare HDF5 files ===
def get_shape(arr):
    return (0,) + arr.shape[1:] if arr.ndim > 1 else (0,)

sample_df = uproot.open(ROOT_FILES[0])["tree"].arrays(library="pd")
sample_data = transform_dataframe(sample_df, MAX_PARTICLES)
dataset_shapes = {k: get_shape(v) for k, v in sample_data.items()}

def create_h5_file(path, shapes):
    f = h5py.File(path, "w")
    dsets = {
        key: f.create_dataset(
            key,
            shape=shape,
            maxshape=(None,) + shape[1:],
            chunks=True,
            compression="gzip",
            compression_opts=4
        ) for key, shape in shapes.items()
    }
    return f, dsets

train_f, train_dsets = create_h5_file(os.path.join(DEST_DIR, "train.h5"), dataset_shapes)
test_f,  test_dsets  = create_h5_file(os.path.join(DEST_DIR, "test.h5"),  dataset_shapes)
val_f,   val_dsets   = create_h5_file(os.path.join(DEST_DIR, "val.h5"),   dataset_shapes)

def append(dset, arr):
    cur = dset.shape[0]
    new = cur + arr.shape[0]
    dset.resize(new, axis=0)
    dset[cur:new] = arr


# === Pass 2: convert and write to HDF5 ===
for file in ROOT_FILES:
    try:
        df = uproot.open(file)["tree"].arrays(library="pd")
        data = transform_dataframe(df, MAX_PARTICLES)
        n_total = common_train + common_test + common_val
        if data["label"].shape[0] < n_total:
            logging.warning(f"Skipping {file}: too few events.")
            continue

        idx = np.arange(data["label"].shape[0])
        np.random.shuffle(idx)
        train_idx = idx[:common_train]
        test_idx  = idx[common_train:common_train + common_test]
        val_idx   = idx[common_train + common_test:n_total]

        for key in data:
            append(train_dsets[key], data[key][train_idx])
            append(test_dsets[key],  data[key][test_idx])
            append(val_dsets[key],   data[key][val_idx])

        logging.info(f"Processed {os.path.basename(file)} with {n_total} events.")
    except Exception as e:
        logging.error(f"Failed to process {file}: {e}")

train_f.close()
test_f.close()
val_f.close()
logging.info("✅ Conversion complete. HDF5 files saved to ParticleNet/Data/")

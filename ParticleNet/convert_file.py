"""
convert_file.py

This script processes a directory of ROOT files (with particle-level jet data)
and converts them into three structured and compressed HDF5 datasets:
train.h5, test.h5, and val.h5.

Input ROOT files:
- Tree name: "tree"
- Columns:
    - part_px, part_py, part_pz, part_energy (jagged lists)
    - label_QCD, label_Hbb, ..., label_Tbl (10 one-hot columns)

Output HDF5 files:
- Saved in PythonProject11/ParticleNet/Data/
- Contains:
    - Jet features: pt, eta, phi, energy, mass
    - Particle features (padded): pt, eta, phi, energy
    - Valid particle count: n_parts
    - Labels: one-hot (shape: N, 10)

The dataset is evenly split across all files and suitable for deep learning models.
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
SOURCE_DIR = os.path.join(PROJECT_DIR, "Data")
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


def pad_event(arr, max_len, pad_value=0.0):
    """Pads a 1D array to fixed length."""
    arr = np.array(arr, dtype=np.float32)
    if arr.shape[0] >= max_len:
        return arr[:max_len]
    pad = np.full((max_len - arr.shape[0],), pad_value, dtype=np.float32)
    return np.concatenate([arr, pad])


def transform_dataframe(df, max_particles=128, eps=1e-6):
    """Transforms a pandas DataFrame to structured numpy arrays."""
    px = torch.tensor(np.stack([pad_event(p, max_particles) for p in df['part_px']]), dtype=torch.float32)
    py = torch.tensor(np.stack([pad_event(p, max_particles) for p in df['part_py']]), dtype=torch.float32)
    pz = torch.tensor(np.stack([pad_event(p, max_particles) for p in df['part_pz']]), dtype=torch.float32)
    E  = torch.tensor(np.stack([pad_event(p, max_particles) for p in df['part_energy']]), dtype=torch.float32)

    mask = (E > 0).float()

    pt = torch.sqrt(px**2 + py**2 + eps)
    p  = torch.sqrt(px**2 + py**2 + pz**2 + eps)
    eta = 0.5 * torch.log((p + pz + eps) / (p - pz + eps))
    phi = torch.atan2(py, px)

    jet_px = (px * mask).sum(dim=1)
    jet_py = (py * mask).sum(dim=1)
    jet_pz = (pz * mask).sum(dim=1)
    jet_e  = (E  * mask).sum(dim=1)

    jet_pt = torch.sqrt(jet_px**2 + jet_py**2 + eps)
    jet_phi = torch.atan2(jet_py, jet_px)
    jet_p = torch.sqrt(jet_px**2 + jet_py**2 + jet_pz**2 + eps)
    jet_eta = 0.5 * torch.log((jet_p + jet_pz + eps) / (jet_p - jet_pz + eps))
    jet_mass = torch.sqrt(torch.clamp(jet_e**2 - (jet_px**2 + jet_py**2 + jet_pz**2), min=0))

    labels = np.stack([df[col].values.astype(int) for col in LABEL_COLS], axis=1)

    return {
        "jet_pt": jet_pt.numpy(),
        "jet_eta": jet_eta.numpy(),
        "jet_phi": jet_phi.numpy(),
        "jet_energy": jet_e.numpy(),
        "jet_mass": jet_mass.numpy(),
        "n_parts": mask.sum(dim=1).numpy(),
        "part_pt": pt.numpy(),
        "part_eta": eta.numpy(),
        "part_phi": phi.numpy(),
        "part_energy": E.numpy(),
        "label": labels
    }


# === First pass: determine common split sizes ===
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


# === Create output HDF5 files ===
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
test_f,  test_dsets  = create_h5_file(os.path.join(DEST_DIR, "test.h5"), dataset_shapes)
val_f,   val_dsets   = create_h5_file(os.path.join(DEST_DIR, "val.h5"), dataset_shapes)

def append(dset, arr):
    cur = dset.shape[0]
    new = cur + arr.shape[0]
    dset.resize(new, axis=0)
    dset[cur:new] = arr


# === Second pass: convert and write to HDF5 ===
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

# === Cleanup ===
train_f.close()
test_f.close()
val_f.close()
logging.info("✅ Conversion complete. HDF5 files saved to ParticleNet/Data/")

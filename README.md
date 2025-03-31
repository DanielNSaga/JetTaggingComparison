# Jet Tagging with ParticleNet and LorentzNet

This repository provides a complete pipeline for comparing two deep learning models — **ParticleNet** and **LorentzNet** — on jet classification using simulated collider data.

The models are trained on a subset of the [JetClass dataset](https://arxiv.org/abs/2206.11898), which contains 100 million jets generated from various Standard Model processes. For this project, we use a curated subset of **10 million jets** (1 million per class) to enable faster training while preserving class balance and physics diversity.

---

## Dataset Details

The original JetClass dataset contains labeled jets from 10 classes:
- QCD
- H→bb, H→cc, H→gg, H→4q, H→qqℓ
- Z→qq, W→qq
- Top→bqq, Top→bl

Each jet contains a variable number of particles (up to ~200). The dataset is stored in `.root` files, one per class. Each particle has information such as 4-momentum, charge, and particle ID flags.

We preprocess the ROOT files and store them in `.h5` format with padding to a fixed number of particles per jet (128). Padding is done per event, and particle-level features are normalized or log-transformed when relevant.

---

## Workflow

1. **Download the data**

   Run the following script to download the ROOT files (10M jets total):
   ```bash
   python download_files.py
   ```

2. **Convert ROOT files to HDF5**

   This step converts variable-length particle arrays to fixed-size tensors, computes derived features, and saves data to `train.h5`, `val.h5`, and `test.h5`.

   Choose either format depending on the model:
   ```bash
   python ParticleNet/convert_file.py
   ```
   or
   ```bash
   python LorentzNet/convert_file.py
   ```

3. **Train a model**

   Once data is prepared, start training:
   ```bash
   python ParticleNet/trainer.py
   ```
   or
   ```bash
   python LorentzNet/trainer.py
   ```

   Both models support logging to TensorBoard, learning rate scheduling with warmup, and automatic model checkpointing.

---

## ParticleNet vs. LorentzNet

| Feature                | ParticleNet                        | LorentzNet                        |
|------------------------|-------------------------------------|-----------------------------------|
| Input format           | Point cloud + per-particle features | 4-vectors + scalar features       |
| Physics priors         | No explicit symmetry                | Lorentz-invariant by construction |
| Architecture           | EdgeConv graph network              | LGEB (Lorentz group equivariant)  |
| Padding                | 128 particles per jet               | 128 particles per jet             |
| Streaming option       | Yes (`stream=True`)                 | Yes (`load_to_ram=True`)          |
| Training time (10M)    | ~4.6 hours                          | ~17 hours                         |
| Accuracy (test set)    | ~70.4%                              | ~73.9%                            |

**ParticleNet** is a performant general-purpose point cloud classifier that operates on local particle neighborhoods. It supports efficient on-disk streaming (`stream=True`), which is helpful when dealing with large datasets.

**LorentzNet** explicitly models the symmetry structure of particle physics by using Lorentz-invariant operations. It can load the full dataset into memory (`load_to_ram=True`), which improves performance at the cost of higher RAM usage.

---

## Configuration Overview

### ParticleNet (config.py)

```python
input_dims = 11
pad_len = 128
num_classes = 10
batch_size = 64
epochs = 10
lr = 1e-3
min_lr = 1e-5
weight_decay = 1e-4
warmup_epochs = 5
patience = 3
stream = True                 # Load batches from disk dynamically
data_format = "channel_last" # Shape: (batch, particles, features)
```

### LorentzNet (config.py)

```python
n_scalar = 7
n_hidden = 128
n_layers = 6
dropout = 0.0
c_weight = 1e-3
batch_size = 64
epochs = 10
lr = 1e-3
min_lr = 1e-5
weight_decay = 1e-4
warmup_epochs = 5
patience = 3
load_to_ram = False           # If True entire dataset is loaded into memory
scheduler_type = "warmup_cosine"
```

---

## Project Structure

```
.
├── Data/                    # Shared preprocessed HDF5 dataset
├── ParticleNet/
│   ├── convert_file.py      # ROOT → HDF5 conversion
│   ├── trainer.py           # Training loop
│   ├── model.py, config.py, ...
├── LorentzNet/
│   ├── convert_file.py
│   ├── trainer.py
│   ├── model.py, config.py, ...
├── download_files.py        # Downloads JetClass subset
├── requirements.txt         # Python dependencies
└── README.md
```

---

## Installation

To install the required Python packages:

```bash
pip install -r requirements.txt
```

This installs:
- PyTorch
- NumPy, h5py
- uproot (for reading ROOT files)
- matplotlib, seaborn
- scikit-learn
- tqdm
- JAX (used optionally for Lorentz-invariant calculations)

---

## Reproducibility

Both models use a `Config` class to manage all training hyperparameters and data paths. Each run saves its full configuration to disk (JSON), ensuring reproducibility and traceability across experiments.

---

## License

This project is licensed under the MIT License.

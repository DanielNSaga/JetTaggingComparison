"""
config.py

Defines the configuration class used to control all aspects of training and evaluation
for the ParticleNet pipeline.

The Config class uses Python's @dataclass to encapsulate hyperparameters, data paths,
logging configuration, and device setup. It also automatically:
- Creates a timestamped run directory for logs and outputs
- Stores the config as a JSON file in the log directory

This ensures reproducibility and makes it easy to manage multiple training runs.
"""

from dataclasses import dataclass, asdict
import torch
import os
from datetime import datetime
import json


@dataclass
class Config:
    """
    Configuration class for managing model, data, training, and logging parameters.

    Attributes:
        input_dims (int): Number of input features per particle.
        num_classes (int): Number of output classes.
        pad_len (int): Number of particles per event after padding.
        stream (bool): Whether to load data on-demand from disk.

        train_path (str): Path to training HDF5 file.
        val_path (str): Path to validation HDF5 file.
        test_path (str): Path to test HDF5 file.
        data_format (str): Feature format, 'channel_first' or 'channel_last'.

        batch_size (int): Number of samples per training batch.
        num_workers (int): Number of parallel data loading workers.

        epochs (int): Number of training epochs.
        lr (float): Initial learning rate.
        min_lr (float): Minimum learning rate after cosine decay.
        weight_decay (float): L2 weight decay.
        warmup_epochs (int): Number of linear warmup epochs.
        patience (int): Early stopping patience or plateau patience.

        device (str): "cuda" or "cpu", auto-selected based on availability.

        run_name (str): Optional run identifier; auto-generated if None.
        log_dir (str): Path to log directory for TensorBoard and saved models.
    """
    # Model
    input_dims: int = 11
    num_classes: int = 10
    pad_len: int = 128
    stream: bool = True

    # Data
    train_path: str = "Data/train.h5"
    val_path: str = "Data/val.h5"
    test_path: str = "Data/test.h5"
    data_format: str = "channel_last"

    batch_size: int = 64
    num_workers: int = 8

    # Training
    epochs: int = 10
    lr: float = 1e-3
    min_lr: float = 1e-5
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    patience: int = 3

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Logging
    run_name: str = None
    log_dir: str = None

    def __post_init__(self):
        """
        Post-initialization hook to:
        - Generate a unique timestamped run name if none is provided.
        - Create a corresponding log directory under 'runs/'.
        - Save the config as JSON for reproducibility.
        """
        if self.run_name is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.run_name = f"run_{timestamp}"

        self.log_dir = os.path.join("runs", self.run_name)
        os.makedirs(self.log_dir, exist_ok=True)

        with open(os.path.join(self.log_dir, "config.json"), "w") as f:
            json.dump(asdict(self), f, indent=4)

"""
config.py

Defines a dataclass-based configuration object for training LorentzNet.
This object holds all hyperparameters and paths needed for training and evaluation.

The class is used to ensure consistency and clarity across the training pipeline.
"""

from dataclasses import dataclass
import torch

@dataclass
class Config:
    """
    Configuration object for LorentzNet training.

    Attributes:
        device (str): Device to use ('cuda' or 'cpu'), automatically selected.
        epochs (int): Number of training epochs.
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of workers for the DataLoader.
        lr (float): Initial learning rate.
        min_lr (float): Minimum learning rate after cosine decay.
        weight_decay (float): Weight decay (L2 regularization).
        warmup_epochs (int): Number of epochs for learning rate warmup.
        patience (int): Patience for ReduceLROnPlateau (early LR reduction).
        load_to_ram (bool): Whether to load the entire dataset into RAM for faster access.

        n_scalar (int): Number of scalar features per node.
        n_hidden (int): Hidden dimension in LorentzNet layers.
        n_class (int): Number of output classes.
        n_layers (int): Number of LGEB layers in LorentzNet.
        dropout (float): Dropout rate used in the final classifier.
        c_weight (float): Weighting factor for coordinate update term.

        data_dir (str): Path to the dataset directory containing .h5 files.

        scheduler_type (str): Type of learning rate scheduler to use. Options:
                              'warmup_cosine', 'cosine', 'step', 'plateau', 'none'
    """
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    epochs: int = 30
    batch_size: int = 64
    num_workers: int = 8
    lr: float = 1e-3
    min_lr: float = 1e-5
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    patience: int = 3
    load_to_ram: bool = False

    # Model
    n_scalar: int = 7
    n_hidden: int = 128
    n_class: int = 10
    n_layers: int = 6
    dropout: float = 0.0
    c_weight: float = 1e-3

    # Data
    data_dir: str = "Data"

    # Scheduler
    scheduler_type: str = "warmup_cosine"

# Config class
from dataclasses import dataclass
import torch
@dataclass
class Config:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    epochs: int = 10
    batch_size: int = 512
    num_workers: int = 8
    load_to_ram: bool = True

    lr: float = 0.01
    weight_decay: float = 0.0

    n_scalar: int = 7
    n_hidden: int = 128
    n_class: int = 10
    n_layers: int = 6
    dropout: float = 0.0
    c_weight: float = 1e-3

    data_dir: str = "Data"
    run_name: str = "lorentznet_a100"
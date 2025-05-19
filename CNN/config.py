from dataclasses import dataclass
import torch
import os
import json
from datetime import datetime


@dataclass
class CNNConfig:

    in_channels: int = 13
    num_classes: int = 10


    train_path: str = "train_splits"
    val_path: str = "DataCNN/val.h5"
    test_path: str = "DataCNN/test.h5"

    batch_size: int = 512
    num_workers: int = 8
    epochs: int = 10
    lr: float = 0.01
    weight_decay: float = 0.0

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Logging / run-info
    run_name: str = None
    data_dir: str = "DataCNN"
    log_dir: str = None

    def __post_init__(self):
        if self.run_name is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.run_name = f"cnn_run_{timestamp}"
        self.run_name = self.run_name.strip()
        self.log_dir = os.path.join("runs", self.run_name)
        os.makedirs(self.log_dir, exist_ok=True)
        # Lagre config til JSON for dokumentasjon
        with open(os.path.join(self.log_dir, "config.json"), "w") as f:
            json.dump(self.__dict__, f, indent=4)

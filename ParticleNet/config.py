from dataclasses import dataclass
import torch
import os
from datetime import datetime
import json

@dataclass
class Config:

    input_dims: int = 11
    num_classes: int = 10
    pad_len: int = 128
    stream: bool = False

    train_path: str = "Data/train.h5"
    val_path: str = "Data/val.h5"
    test_path: str = "Data/test.h5"
    data_format: str = "channel_last"

    batch_size: int = 512
    num_workers: int = 8
    epochs: int = 10
    lr: float = 0.01
    weight_decay: float = 0.0

    # Device:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    run_name: str = None
    data_dir: str = "Data"

    def __post_init__(self):
        if self.run_name is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.run_name = f"particlenet_a100_{timestamp}"
        self.run_name = self.run_name.strip()
        self.log_dir = os.path.join("runs", self.run_name)
        os.makedirs(self.log_dir, exist_ok=True)
        with open(os.path.join(self.log_dir, "config.json"), "w") as f:
            json.dump(self.__dict__, f, indent=4)

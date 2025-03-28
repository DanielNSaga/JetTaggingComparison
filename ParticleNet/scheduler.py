"""
scheduler.py

This module defines a custom learning rate scheduler class for PyTorch training loops.

The CosineWithWarmupScheduler supports:
- Linear warmup over a specified number of epochs
- Cosine decay from base learning rate to a minimum learning rate
- Integration with ReduceLROnPlateau as a fallback
- Tracking and plotting of learning rate history per epoch

Usage:
    scheduler = CosineWithWarmupScheduler(optimizer, warmup_epochs, total_epochs, min_lr)
    for epoch in range(total_epochs):
        scheduler.step(epoch, val_loss)
        ...
    scheduler.plot()
"""

import math
import torch
import matplotlib.pyplot as plt
import os


class CosineWithWarmupScheduler:
    """
    Scheduler combining linear warmup and cosine annealing with optional ReduceLROnPlateau fallback.

    Args:
        optimizer (Optimizer): The optimizer whose learning rate will be scheduled.
        warmup_epochs (int): Number of epochs to linearly increase learning rate.
        total_epochs (int): Total number of epochs for the schedule.
        min_lr (float): The minimum learning rate after cosine decay.
        log_dir (str): Optional directory to save LR plots. Defaults to current directory.
        run_name (str): Name prefix for output files.
    """
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-5, log_dir=None, run_name="run"):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.lr_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, min_lr=min_lr, verbose=False
        )

        self.history = []
        self.log_dir = log_dir or "."
        self.run_name = run_name

    def step(self, epoch, val_loss=None):
        """
        Updates the learning rate for the current epoch and optionally applies ReduceLROnPlateau.

        Args:
            epoch (int): Current epoch index.
            val_loss (float, optional): Validation loss used for plateau scheduling.

        Returns:
            float: The updated learning rate.
        """
        if epoch < self.warmup_epochs:
            scale = (epoch + 1) / self.warmup_epochs
            lrs = [lr * scale for lr in self.base_lrs]
        else:
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            scale = 0.5 * (1 + math.cos(math.pi * progress))
            lrs = [self.min_lr + (base_lr - self.min_lr) * scale for base_lr in self.base_lrs]

        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group["lr"] = lr

        if val_loss is not None:
            self.lr_plateau.step(val_loss)

        self.history.append(lrs[0])
        return lrs[0]

    def plot(self):
        """
        Plots and saves the learning rate schedule as a PNG file.

        The file is saved to `log_dir/run_name_lr_schedule.png`.
        """
        if not self.history:
            print("No learning rate history to plot.")
            return

        plt.figure(figsize=(8, 4))
        plt.plot(self.history, label="Learning Rate")
        plt.xlabel("Epoch")
        plt.ylabel("LR")
        plt.title("Learning Rate Schedule")
        plt.grid(True)
        plt.tight_layout()

        out_path = os.path.join(self.log_dir, f"{self.run_name}_lr_schedule.png")
        plt.savefig(out_path)
        print(f"Saved LR schedule to {out_path}")

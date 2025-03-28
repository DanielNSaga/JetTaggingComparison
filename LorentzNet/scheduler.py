"""
scheduler.py

Implements a learning rate scheduler combining linear warmup and cosine annealing,
with optional fallback to ReduceLROnPlateau. Also supports plotting the learning rate
curve after training.

This scheduler is designed for modern deep learning tasks with long training schedules,
where a warmup period followed by cosine decay is beneficial.

Typical usage:
    scheduler = CosineWithWarmupScheduler(optimizer, warmup_epochs=5, total_epochs=50)
    for epoch in range(epochs):
        ...
        scheduler.step(epoch, val_loss)
    scheduler.plot()
"""

import math
import torch
import matplotlib.pyplot as plt

class CosineWithWarmupScheduler:
    """
    Custom learning rate scheduler with linear warmup and cosine decay.

    Also integrates ReduceLROnPlateau as a fallback in case of validation loss plateauing.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer whose LR will be updated.
        warmup_epochs (int): Number of initial epochs with linearly increasing LR.
        total_epochs (int): Total number of training epochs.
        min_lr (float): The minimum learning rate at the end of cosine decay.
    """
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-5):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

        # Fallback scheduler in case of plateau
        self.lr_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, min_lr=min_lr, verbose=False
        )

        self.history = []  # Store LR per epoch for plotting

    def step(self, epoch, val_loss=None):
        """
        Update the learning rate for the current epoch.

        Args:
            epoch (int): The current epoch number (0-based).
            val_loss (float, optional): Validation loss used for ReduceLROnPlateau.

        Returns:
            float: The learning rate applied this epoch.
        """
        if epoch < self.warmup_epochs:
            # Linear warmup
            scale = (epoch + 1) / self.warmup_epochs
            lrs = [lr * scale for lr in self.base_lrs]
        else:
            # Cosine annealing after warmup
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            scale = 0.5 * (1 + math.cos(math.pi * progress))
            lrs = [self.min_lr + (base_lr - self.min_lr) * scale for base_lr in self.base_lrs]

        for group, lr in zip(self.optimizer.param_groups, lrs):
            group["lr"] = lr

        if val_loss is not None:
            self.lr_plateau.step(val_loss)

        self.history.append(lrs[0])
        return lrs[0]

    def plot(self, filename="lr_schedule.png"):
        """
        Plot and save the learning rate schedule.

        Args:
            filename (str): Path to save the plot image.
        """
        if not self.history:
            print("No LR history to plot.")
            return
        plt.figure()
        plt.plot(self.history, label="LR")
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate Schedule")
        plt.grid()
        plt.tight_layout()
        plt.savefig(filename)
        print(f"Saved LR plot to {filename}")

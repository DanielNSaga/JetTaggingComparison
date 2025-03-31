"""
trainer.py

This module defines a Trainer class for training the ParticleNet model on jet tagging data.
It includes:
- Mixed precision training with `torch.amp`
- Learning rate scheduling with warmup and cosine decay
- TensorBoard logging of training/validation metrics
- Confusion matrix visualization and classification reports
- Saving best model based on validation accuracy
- Logging all progress to a text file

Expected structure:
- Uses `Config` for hyperparameters and paths
- Loads HDF5 datasets via `get_datasets`
- Logs to `runs/<run_name>` using TensorBoard and log files
"""

import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix, classification_report

import matplotlib.pyplot as plt
import seaborn as sns
from config import Config
from model import ParticleNet
from dataset import get_datasets
from scheduler import CosineWithWarmupScheduler


class Trainer:
    """
    Training engine for ParticleNet.

    Handles training loop, validation, logging, scheduling, and saving.

    Args:
        cfg (Config): A configuration dataclass containing all training parameters.
    """
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.model = ParticleNet(cfg.input_dims, cfg.num_classes).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.criterion = nn.BCEWithLogitsLoss()
        self.scheduler = CosineWithWarmupScheduler(
            self.optimizer,
            warmup_epochs=cfg.warmup_epochs,
            total_epochs=cfg.epochs,
            min_lr=cfg.min_lr,
            log_dir=cfg.log_dir,
            run_name=cfg.run_name
        )

        self.scaler = torch.amp.GradScaler(enabled=(cfg.device == "cuda"))

        train_ds, val_ds, _ = get_datasets(
            cfg.train_path, cfg.val_path, cfg.test_path,
            pad_len=cfg.pad_len, data_format=cfg.data_format, stream=cfg.stream
        )
        self.train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                                       num_workers=cfg.num_workers, pin_memory=True)
        self.val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                                     num_workers=cfg.num_workers, pin_memory=True)

        self.run_dir = os.path.join("runs", cfg.run_name)
        os.makedirs(self.run_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.run_dir)
        self.log_file = open(os.path.join(self.run_dir, "training.log"), "w")
        self.log_file.write(f"Device used: {cfg.device}\n")

        with open(os.path.join(self.run_dir, "config.json"), "w") as f:
            json.dump(cfg.__dict__, f, indent=2)

    def _to_tensor(self, x):
        """Converts numpy array to tensor if not already a tensor."""
        return x if isinstance(x, torch.Tensor) else torch.from_numpy(x)

    def train_epoch(self, epoch):
        """
        Executes one full training epoch.

        Args:
            epoch (int): Current epoch number.

        Returns:
            float: Average training loss.
            float: Training accuracy.
        """
        self.model.train()
        total_loss, correct, total = 0, 0, 0

        with tqdm(self.train_loader, desc=f"Train {epoch}", leave=False) as pbar:
            for batch in pbar:
                X, y = batch["X"], batch["y"]
                points = self._to_tensor(X["points"]).float().to(self.device)
                features = self._to_tensor(X["features"]).float().to(self.device)
                mask = (features.abs().sum(dim=2, keepdim=True) != 0).float()
                labels = self._to_tensor(y).float().to(self.device)

                self.optimizer.zero_grad()
                with torch.amp.autocast(device_type=self.cfg.device):
                    outputs = self.model(points, features, mask)
                    loss = self.criterion(outputs, labels)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                total_loss += loss.item() * labels.size(0)
                preds = outputs.argmax(dim=1)
                targets = labels.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += labels.size(0)


                running_loss = total_loss / total
                running_acc = correct / total
                pbar.set_postfix({
                    "loss": f"{running_loss:.4f}",
                    "acc": f"{running_acc:.4f}"
                })

        epoch_loss = total_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def validate(self, epoch):
        """
        Runs evaluation on the validation set.

        Args:
            epoch (int): Current epoch number.

        Returns:
            float: Average validation loss.
            float: Validation accuracy.
            np.ndarray: Confusion matrix.
            dict: Classification report.
        """
        self.model.eval()
        total_loss, correct, total = 0, 0, 0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"Val {epoch}", leave=False):
                X, y = batch["X"], batch["y"]
                points = self._to_tensor(X["points"]).float().to(self.device)
                features = self._to_tensor(X["features"]).float().to(self.device)
                mask = (features.abs().sum(dim=2, keepdim=True) != 0).float()
                labels = self._to_tensor(y).float().to(self.device)

                with torch.amp.autocast(device_type=self.cfg.device):
                    outputs = self.model(points, features, mask)
                    loss = self.criterion(outputs, labels)

                total_loss += loss.item() * labels.size(0)
                preds = outputs.argmax(dim=1)
                targets = labels.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += labels.size(0)

                all_preds.append(preds.cpu())
                all_targets.append(targets.cpu())

        all_preds = torch.cat(all_preds).numpy()
        all_targets = torch.cat(all_targets).numpy()
        cm = confusion_matrix(all_targets, all_preds)
        report = classification_report(all_targets, all_preds, output_dict=True)

        return total_loss / total, correct / total, cm, report

    def train(self):
        """
        Full training loop including logging, saving best model, and scheduler stepping.
        """
        best_acc = 0
        for epoch in range(self.cfg.epochs):
            start = time.time()
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc, cm, report = self.validate(epoch)
            lr = self.scheduler.step(epoch, val_loss)
            epoch_time = time.time() - start

            # Log scalar metrics
            self.writer.add_scalar("LR", lr, epoch)
            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("Loss/val", val_loss, epoch)
            self.writer.add_scalar("Acc/train", train_acc, epoch)
            self.writer.add_scalar("Acc/val", val_acc, epoch)

            # Log to file
            log_line = (f"Epoch {epoch+1:03d}: LR={lr:.2e} | "
                        f"Train Loss={train_loss:.4f}, Acc={train_acc:.4f} | "
                        f"Val Loss={val_loss:.4f}, Acc={val_acc:.4f} | "
                        f"Time: {epoch_time:.2f}s\n")
            print(log_line.strip())
            self.log_file.write(log_line)

            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), os.path.join(self.run_dir, "best_model.pt"))

            # Log confusion matrix figure
            self.writer.add_figure("Confusion Matrix", self._plot_confusion_matrix(cm), epoch)

        self.writer.close()
        self.log_file.write("Training complete.\n")
        self.log_file.close()

    def _plot_confusion_matrix(self, cm):
        """
        Plots and returns a matplotlib figure of the confusion matrix.

        Args:
            cm (np.ndarray): Confusion matrix to plot.

        Returns:
            matplotlib.figure.Figure: The plotted figure.
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")
        return fig


if __name__ == "__main__":
    cfg = Config()
    trainer = Trainer(cfg)
    trainer.train()

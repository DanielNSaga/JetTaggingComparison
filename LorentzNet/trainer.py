"""
trainer.py

Training pipeline for the LorentzNet model for jet tagging.

This module provides a fully-featured training engine with:
- Mixed precision training using torch.amp
- Custom learning rate scheduler with warmup and cosine decay
- Accuracy and loss tracking
- Confusion matrix logging (per epoch) to TensorBoard
- Model checkpointing (best validation accuracy)
- Runtime tracking and LR schedule plotting
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from model import LorentzNet
from dataset import retrieve_dataloaders
from config import Config
from scheduler import CosineWithWarmupScheduler

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')


class Trainer:
    """
    Trainer class for LorentzNet model.

    Args:
        cfg (Config): Configuration object with model, data, and training settings.
    """
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.model = LorentzNet(
            cfg.n_scalar, cfg.n_hidden, cfg.n_class, cfg.n_layers, cfg.c_weight, cfg.dropout
        ).to(self.device)

        if hasattr(torch, "compile"):
            if torch.device("cuda"):
                self.model = torch.compile(self.model)

        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = CosineWithWarmupScheduler(self.optimizer, cfg.warmup_epochs, cfg.epochs, cfg.min_lr)
        self.scaler = torch.amp.GradScaler(enabled=cfg.device == "cuda")

        _, loaders = retrieve_dataloaders(cfg.batch_size, cfg.data_dir, cfg.num_workers, cfg.load_to_ram)
        self.train_loader = loaders["train"]
        self.val_loader = loaders["val"]

        logdir = os.path.join("runs", "lorentznet", datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.writer = SummaryWriter(logdir)

        logging.info(f"Device: {cfg.device}")
        logging.info(f"Model: LorentzNet, Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def _forward(self, batch):
        """
        Prepares input batch and moves all tensors to the correct device.

        Args:
            batch (tuple): A batch from the dataloader.

        Returns:
            Tuple of model inputs: scalars, p4s, edges, atom_mask, edge_mask, labels, N
        """
        labels, p4s, scalars, atom_mask, edge_mask, edges = batch
        N = p4s.shape[1]

        scalars = scalars.view(-1, scalars.shape[-1]).to(self.device).float()
        p4s = p4s.view(-1, p4s.shape[-1]).to(self.device).float()
        labels = labels.to(self.device).long()
        atom_mask = atom_mask.to(self.device)
        edge_mask = edge_mask.to(self.device)
        edges = (edges[0].to(self.device), edges[1].to(self.device))
        return scalars, p4s, edges, atom_mask, edge_mask, labels, N

    def _epoch(self, loader, training=True):
        """
        Runs one epoch of training or evaluation.

        Args:
            loader (DataLoader): The dataloader (train or val).
            training (bool): Whether to train (True) or evaluate (False).

        Returns:
            Tuple: (avg_loss, accuracy, confusion_matrix)
        """
        self.model.train() if training else self.model.eval()
        total_loss, correct, total = 0, 0, 0
        all_preds, all_targets = [], []

        context = torch.enable_grad() if training else torch.no_grad()
        desc = "Train" if training else "Val"

        with context:
            for batch in tqdm(loader, desc=desc, leave=False):
                scalars, p4s, edges, atom_mask, edge_mask, labels, N = self._forward(batch)
                self.optimizer.zero_grad()

                with torch.amp.autocast(device_type=self.cfg.device):
                    outputs = self.model(scalars, p4s, edges, atom_mask, edge_mask, N)
                    loss = self.criterion(outputs, labels)

                if training:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                preds = outputs.argmax(dim=1)
                total_loss += loss.item()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                all_preds.append(preds.cpu())
                all_targets.append(labels.cpu())

        acc = correct / total
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        cm = confusion_matrix(all_targets, all_preds, labels=torch.arange(self.cfg.n_class))
        return total_loss / len(loader), acc, cm

    def _log_confusion_matrix(self, cm, epoch, phase):
        """
        Logs a confusion matrix to TensorBoard.

        Args:
            cm (np.ndarray): Confusion matrix.
            epoch (int): Current epoch index.
            phase (str): "Train" or "Val".
        """
        fig, ax = plt.subplots(figsize=(6, 6))
        disp = ConfusionMatrixDisplay(cm)
        disp.plot(ax=ax)
        plt.title(f"{phase} Confusion Matrix - Epoch {epoch+1}")
        self.writer.add_figure(f"{phase}/ConfusionMatrix", fig, epoch)
        plt.close(fig)

    def train(self):
        """
        Full training loop across all epochs.

        - Logs losses, accuracies, learning rates, confusion matrices
        - Tracks best validation accuracy
        - Saves best model to file
        - Records total training time
        - Logs everything to TensorBoard
        """
        best_acc = 0
        start = time.time()

        for epoch in range(self.cfg.epochs):
            train_loss, train_acc, train_cm = self._epoch(self.train_loader, training=True)
            val_loss, val_acc, val_cm = self._epoch(self.val_loader, training=False)
            lr = self.scheduler.step(epoch, val_loss)

            # TensorBoard logging
            self.writer.add_scalar("Loss/Train", train_loss, epoch)
            self.writer.add_scalar("Loss/Val", val_loss, epoch)
            self.writer.add_scalar("Acc/Train", train_acc, epoch)
            self.writer.add_scalar("Acc/Val", val_acc, epoch)
            self.writer.add_scalar("LR", lr, epoch)
            self._log_confusion_matrix(train_cm, epoch, "Train")
            self._log_confusion_matrix(val_cm, epoch, "Val")

            logging.info(f"Epoch {epoch+1:03d} | LR: {lr:.2e} | "
                         f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                         f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), "best_model.pt")
                logging.info("Saved new best model.")

        elapsed = time.time() - start
        logging.info(f"Training complete in {elapsed/60:.2f} minutes.")
        self.scheduler.plot("lorentznet_lr.png")
        self.writer.close()


if __name__ == "__main__":
    cfg = Config()
    trainer = Trainer(cfg)
    trainer.train()

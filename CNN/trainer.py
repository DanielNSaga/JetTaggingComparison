import os
import random
import time
import glob
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch_optimizer import RAdam, Lookahead
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from config import CNNConfig
from model import JetImageCNN
from dataset import JetImageDatasetWrapper

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True

class CNNTrainer:
    def __init__(self, cfg: CNNConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        model = JetImageCNN(cfg.in_channels, cfg.num_classes).to(self.device)
        self.model = torch.compile(model, mode="reduce-overhead")

        radam = RAdam(
            self.model.parameters(),
            lr=cfg.lr,
            betas=(0.95, 0.999),
            eps=1e-5,
            weight_decay=cfg.weight_decay
        )
        self.optimizer = Lookahead(radam, k=6, alpha=0.5)

        self.criterion = nn.CrossEntropyLoss()
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.device.type == "cuda"))

        self.log_file = open(os.path.join(cfg.log_dir, "training.log"), "w")
        self.log_file.write(f"Device used: {cfg.device}\n")
        self.writer = SummaryWriter(log_dir=cfg.log_dir)

        self.train_chunk_files = sorted(glob.glob(os.path.join(cfg.train_path, "train_part*.h5")))
        if not self.train_chunk_files:
            raise RuntimeError(f"No HDF5 chunk files found in {cfg.train_path}")

        self.val_loader = JetImageDatasetWrapper(
            cfg.val_path,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            shuffle=False
        )

        self.global_step = 0
        self.best_acc = 0.0
        self.total_iterations = (len(self.train_chunk_files) // 2) * cfg.epochs
        self.switch_iter = int(self.total_iterations * 0.7)
        self.decay_steps = 2500
        self.final_lr = cfg.lr * 0.01

    def _adjust_lr(self):
        if self.global_step < self.switch_iter:
            lr = self.cfg.lr
        else:
            decay_iter = self.global_step - self.switch_iter
            factor = decay_iter // self.decay_steps
            decay_ratio = (self.final_lr / self.cfg.lr) ** (
                factor * self.decay_steps / (self.total_iterations - self.switch_iter)
            )
            lr = self.cfg.lr * decay_ratio
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    def _log_confmat(self, cm, epoch, name):
        fig, ax = plt.subplots(figsize=(6, 6))
        disp = ConfusionMatrixDisplay(cm)
        disp.plot(ax=ax)
        ax.set_title(f"{name} Confusion Matrix (Epoch {epoch})")
        self.writer.add_figure(f"{name}/ConfusionMatrix", fig, epoch)
        plt.close(fig)

    def _run_epoch(self, epoch, training=True):
        if training:
            self.model.train()
            random.shuffle(self.train_chunk_files)
            chunk_files = self.train_chunk_files
        else:
            self.model.eval()
            chunk_files = [self.cfg.val_path]

        total_loss = 0.0
        correct = 0
        total = 0

        all_preds = []
        all_targets = []


        pairs = [chunk_files[i : i+2] for i in range(0, len(chunk_files), 2)]
        for pair in pairs:
            for chunk_file in pair:
                loader = JetImageDatasetWrapper(
                    chunk_file,
                    batch_size=self.cfg.batch_size,
                    num_workers=self.cfg.num_workers,
                    shuffle=training
                )
                with (torch.enable_grad() if training else torch.no_grad()):
                    for images, labels in tqdm(loader, desc=("Train" if training else "Val"), leave=False):
                        images, labels = images.to(self.device), labels.to(self.device)

                        if training:
                            self.optimizer.zero_grad(set_to_none=True)

                        with torch.amp.autocast(device_type=self.cfg.device, dtype=torch.float16):
                            outputs = self.model(images)
                            loss = self.criterion(outputs, labels.argmax(dim=1))

                        if training:
                            self.scaler.scale(loss).backward()
                            self.scaler.step(self.optimizer)
                            self.scaler.update()

                            lr = self._adjust_lr()
                            self.writer.add_scalar("LR", lr, self.global_step)
                            self.global_step += 1

                        total_loss += loss.item() * labels.size(0)
                        preds = outputs.argmax(dim=1)
                        targs = labels.argmax(dim=1)
                        correct += (preds == targs).sum().item()
                        total += labels.size(0)

                        all_preds.append(preds.cpu())
                        all_targets.append(targs.cpu())

        avg_loss = total_loss / max(total, 1)
        acc = correct / max(total, 1)

        if all_preds:
            all_preds = torch.cat(all_preds)
            all_targets = torch.cat(all_targets)
            cm = confusion_matrix(all_targets, all_preds, labels=range(self.cfg.num_classes))
        else:
            cm = None

        return avg_loss, acc, cm

    def train(self):
        start_time = time.time()
        for epoch in range(self.cfg.epochs):
            # 1) Train
            train_loss, train_acc, train_cm = self._run_epoch(epoch, training=True)

            # 2) Val
            val_loss, val_acc, val_cm = self._run_epoch(epoch, training=False)

            # 3) Logg
            self.writer.add_scalar("Loss/Train", train_loss, epoch)
            self.writer.add_scalar("Acc/Train", train_acc, epoch)
            self.writer.add_scalar("Loss/Val", val_loss, epoch)
            self.writer.add_scalar("Acc/Val", val_acc, epoch)

            if train_cm is not None:
                self._log_confmat(train_cm, epoch, "Train")
            if val_cm is not None:
                self._log_confmat(val_cm, epoch, "Val")

            msg = (f"[Epoch {epoch+1:02d}] "
                   f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                   f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            print(msg)
            self.log_file.write(msg + "\n")

            if val_acc > self.best_acc:
                self.best_acc = val_acc
                torch.save(self.model.state_dict(), os.path.join(self.cfg.log_dir, "best_model.pt"))
                print("✅ New best model saved.")

        total_mins = (time.time() - start_time) / 60.0
        print(f"⏱️ Training complete in {total_mins:.1f} minutes")
        self.log_file.write("Training complete.\n")
        self.log_file.close()
        self.writer.close()

if __name__ == "__main__":
    cfg = CNNConfig()
    trainer = CNNTrainer(cfg)
    trainer.train()

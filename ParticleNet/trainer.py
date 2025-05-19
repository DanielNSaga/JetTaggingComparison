import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter
from torch_optimizer import RAdam, Lookahead

from config import Config
from model import ParticleNet
from dataset import get_datasets

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True

class Trainer:

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        model = ParticleNet(cfg.input_dims, cfg.num_classes).to(self.device)
        self.model = torch.compile(model, mode="reduce-overhead")

        radam = RAdam(self.model.parameters(), lr=cfg.lr, betas=(0.95, 0.999), eps=1e-5, weight_decay=cfg.weight_decay)
        self.optimizer = Lookahead(radam, k=6, alpha=0.5)

        self.criterion = nn.BCEWithLogitsLoss()
        self.scaler = torch.amp.GradScaler(enabled=True)

        train_ds, val_ds, _ = get_datasets(
            cfg.train_path, cfg.val_path, cfg.test_path,
            pad_len=cfg.pad_len, data_format=cfg.data_format, stream=cfg.stream
        )
        self.train_loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                                                          num_workers=cfg.num_workers, pin_memory=True)
        self.val_loader   = torch.utils.data.DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                                                          num_workers=cfg.num_workers, pin_memory=True)

        self.run_dir = os.path.join("runs", cfg.run_name)
        os.makedirs(self.run_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.run_dir)
        self.log_file = open(os.path.join(self.run_dir, "training.log"), "w")
        self.log_file.write(f"Device used: {cfg.device}\n")

        import json
        with open(os.path.join(self.run_dir, "config.json"), "w") as f:
            json.dump(cfg.__dict__, f, indent=2)

        self.total_iterations = len(self.train_loader) * cfg.epochs
        self.switch_iter = int(self.total_iterations * 0.7)
        self.decay_steps = 2500
        self.final_lr = cfg.lr * 0.01
        self.global_step = 0

    def _to_tensor(self, x):
        return x if isinstance(x, torch.Tensor) else torch.tensor(x)

    def _prepare_batch(self, batch):

        X, y = batch["X"], batch["y"]
        points   = self._to_tensor(X["points"]).float().to(self.device, non_blocking=True)
        features = self._to_tensor(X["features"]).float().to(self.device, non_blocking=True)
        mask = (features.abs().sum(dim=2, keepdim=True) != 0).float().to(self.device, non_blocking=True)
        labels = self._to_tensor(y).float().to(self.device, non_blocking=True)
        return points, features, mask, labels

    def _adjust_lr(self):
        if self.global_step < self.switch_iter:
            lr = self.cfg.lr
        else:
            decay_iter = self.global_step - self.switch_iter
            factor = decay_iter // self.decay_steps
            decay_ratio = (self.final_lr / self.cfg.lr) ** (factor * self.decay_steps / (self.total_iterations - self.switch_iter))
            lr = self.cfg.lr * decay_ratio
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def _run_epoch(self, loader, training=True):
        self.model.train() if training else self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        all_preds, all_targets = [], []
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for batch in tqdm(loader, desc="Train" if training else "Val", leave=False):
                points, features, mask, labels = self._prepare_batch(batch)

                self.optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=self.cfg.device, dtype=torch.float16):
                    outputs = self.model(points, features, mask)
                    loss = self.criterion(outputs, labels)
                if training:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    lr = self._adjust_lr()
                    self.writer.add_scalar("LR", lr, self.global_step)
                    self.global_step += 1

                total_loss += loss.item() * labels.size(0)
                preds = outputs.argmax(dim=1)
                targets = labels.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += labels.size(0)
                all_preds.append(preds.cpu())
                all_targets.append(targets.cpu())

        acc = correct / total
        cm = confusion_matrix(torch.cat(all_targets), torch.cat(all_preds), labels=list(range(self.cfg.num_classes)))
        return total_loss / total, acc, cm

    def _log_confmat(self, cm, epoch, name):
        fig, ax = plt.subplots(figsize=(6, 6))
        disp = ConfusionMatrixDisplay(cm)
        disp.plot(ax=ax)
        ax.set_title(f"{name} Confusion Matrix (Epoch {epoch})")
        self.writer.add_figure(f"{name}/ConfusionMatrix", fig, epoch)
        plt.close(fig)

    def train(self):
        start = time.time()
        best_acc = 0.0
        for epoch in range(self.cfg.epochs):
            train_loss, train_acc, train_cm = self._run_epoch(self.train_loader, training=True)
            val_loss, val_acc, val_cm = self._run_epoch(self.val_loader, training=False)

            self.writer.add_scalar("Loss/Train", train_loss, epoch)
            self.writer.add_scalar("Loss/Val", val_loss, epoch)
            self.writer.add_scalar("Acc/Train", train_acc, epoch)
            self.writer.add_scalar("Acc/Val", val_acc, epoch)

            self._log_confmat(train_cm, epoch, "Train")
            self._log_confmat(val_cm, epoch, "Val")

            log_line = (f"[Epoch {epoch+1:02d}] Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                        f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}\n")
            print(log_line.strip())
            self.log_file.write(log_line)

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), os.path.join(self.run_dir, "best_model.pt"))
                print("✅ New best model saved.")

        elapsed = time.time() - start
        print(f"⏱️ Training complete in {elapsed/60:.1f} minutes")
        self.writer.close()
        self.log_file.write("Training complete.\n")
        self.log_file.close()


if __name__ == "__main__":
    cfg = Config()
    trainer = Trainer(cfg)
    trainer.train()

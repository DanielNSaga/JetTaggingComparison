import os
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch_optimizer import RAdam, Lookahead

from model import LorentzNet
from dataset import retrieve_dataloaders
from config import Config

# Optimalisering for A100
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch._inductor.config.triton.cudagraph_skip_dynamic_graphs=True

class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        base_model = LorentzNet(cfg.n_scalar, cfg.n_hidden, cfg.n_class, cfg.n_layers, cfg.c_weight, cfg.dropout).to(self.device)
        print("Compiling model with torch.compile using max-autotune mode...")
        self.model = torch.compile(base_model, mode="max-autotune")
        print("Model compiled successfully with max-autotune.")

        radam = RAdam(self.model.parameters(), lr=cfg.lr, betas=(0.95, 0.999), eps=1e-5, weight_decay=0.0)
        self.optimizer = Lookahead(radam, k=6, alpha=0.5)

        self.criterion = nn.CrossEntropyLoss()
        self.scaler = torch.amp.GradScaler(enabled=True)

        _, loaders = retrieve_dataloaders(cfg.batch_size, cfg.data_dir, cfg.num_workers, cfg.load_to_ram, pin_memory=True)
        self.train_loader = loaders["train"]
        self.val_loader = loaders["val"]

        self.writer = SummaryWriter(log_dir=os.path.join("runs", cfg.run_name or datetime.now().strftime("%Y%m%d-%H%M%S")))
        self.best_acc = 0

        self.total_iterations = len(self.train_loader) * cfg.epochs
        self.switch_iter = int(self.total_iterations * 0.7)
        self.decay_steps = 2500
        self.final_lr = cfg.lr * 0.01
        self.global_step = 0

        self.max_grad_norm = 1.0

    def _prepare_batch(self, batch):
        labels, p4s, scalars, atom_mask, edge_mask, edges = batch
        B, N = scalars.shape[0], scalars.shape[1]

        scalars = scalars.view(B * N, -1).to(self.device, non_blocking=True).float()
        p4s = p4s.view(B * N, -1).to(self.device, non_blocking=True).float()

        labels = labels.to(self.device, non_blocking=True).long()
        atom_mask = atom_mask.to(self.device, non_blocking=True)
        edge_mask = edge_mask.to(self.device, non_blocking=True)
        edges = (edges[0].to(self.device, non_blocking=True), edges[1].to(self.device, non_blocking=True))

        return scalars, p4s, edges, atom_mask, edge_mask, labels, N

    def _adjust_lr(self):
        if self.global_step < self.switch_iter:
            lr = self.cfg.lr
        else:
            decay_iter = self.global_step - self.switch_iter
            factor = decay_iter // self.decay_steps
            decay_ratio = (self.final_lr / self.cfg.lr) ** (factor * self.decay_steps / (self.total_iterations - self.switch_iter))
            lr = self.cfg.lr * decay_ratio

        lr = max(lr, self.final_lr)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def _run_epoch(self, loader, training=True):
        self.model.train() if training else self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        all_preds, all_targets = [], []
        desc = "Train" if training else "Val"
        pbar = tqdm(loader, desc=desc, leave=False)

        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for batch in pbar:
                scalars, p4s, edges, atom_mask, edge_mask, labels, N = self._prepare_batch(batch)

                self.optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=self.cfg.device, dtype=torch.float16):
                    logits = self.model(scalars, p4s, edges, atom_mask, edge_mask, N)
                    loss = self.criterion(logits, labels)

                if training:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    lr = self._adjust_lr()
                    self.writer.add_scalar("LR", lr, self.global_step)
                    self.global_step += 1

                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                total_loss += loss.item()
                all_preds.append(preds.cpu())
                all_targets.append(labels.cpu())

                avg_loss = total_loss / (len(all_preds))
                avg_acc = correct / total if total > 0 else 0
                pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc:.4f}")

        acc = correct / total
        cm = confusion_matrix(torch.cat(all_targets), torch.cat(all_preds), labels=torch.arange(self.cfg.n_class))
        return total_loss / len(loader), acc, cm

    def _log_confmat(self, cm, epoch, name):
        fig, ax = plt.subplots(figsize=(6, 6))
        disp = ConfusionMatrixDisplay(cm)
        disp.plot(ax=ax)
        plt.title(f"{name} Confusion Matrix (Epoch {epoch})")
        self.writer.add_figure(f"{name}/ConfusionMatrix", fig, epoch)
        plt.close(fig)

    def train(self):
        start = time.time()
        print("Training started...")
        self.writer.add_text("Info", "Training started with max-autotune mode, gradient clipping and detailed logging", 0)

        for epoch in range(self.cfg.epochs):
            print(f"Epoch {epoch+1}/{self.cfg.epochs} in progress...")
            train_loss, train_acc, train_cm = self._run_epoch(self.train_loader, training=True)
            val_loss, val_acc, val_cm = self._run_epoch(self.val_loader, training=False)

            self.writer.add_scalar("Loss/Train", train_loss, epoch)
            self.writer.add_scalar("Loss/Val", val_loss, epoch)
            self.writer.add_scalar("Acc/Train", train_acc, epoch)
            self.writer.add_scalar("Acc/Val", val_acc, epoch)

            self._log_confmat(train_cm, epoch, "Train")
            self._log_confmat(val_cm, epoch, "Val")

            print(f"[Epoch {epoch+1:02d}] Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

            if val_acc > self.best_acc:
                self.best_acc = val_acc
                torch.save(self.model.state_dict(), "best_model.pt")
                print("✅ New best model saved.")
                self.writer.add_text("Info", f"Epoch {epoch+1}: New best model saved with val acc: {val_acc:.4f}", epoch)

        elapsed = time.time() - start
        print(f"⏱️ Training complete in {elapsed / 60:.1f} minutes")
        self.writer.add_text("Info", f"Training complete in {elapsed / 60:.1f} minutes", self.cfg.epochs)
        self.writer.close()

if __name__ == "__main__":
    cfg = Config()
    trainer = Trainer(cfg)
    trainer.train()

import os
import time
import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
from particlenet import ParticleNet
from args import Args
from tqdm import tqdm

# Initialiser args og device
args = Args()
DEVICE = args.device


# Dataset-definisjon – bruk weights_only=False
class ProcessedDataset(torch_geometric.data.InMemoryDataset):
    def __init__(self, dataset_path):
        super().__init__(dataset_path)
        self.data, self.slices = torch.load(os.path.join(dataset_path, "processed", "data.pt"), weights_only=False)

    @property
    def processed_file_names(self):
        return ["data.pt"]


# Treningsløkke med progress bar og ekstra metrikker
def train_epoch(model, optimizer, train_loader, epoch):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_train_preds = []
    all_train_labels = []
    start_time = time.time()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=True)
    for batch in pbar:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(batch)
        targets = batch.y.long()
        loss = F.cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        preds = outputs.argmax(dim=1)
        total_correct += (preds == targets).sum().item()
        total_samples += batch_size

        # Akkumuler for metrikker
        all_train_preds.extend(preds.cpu().numpy())
        all_train_labels.extend(targets.cpu().numpy())

        cur_loss = total_loss / total_samples
        cur_acc = total_correct / total_samples
        current_iter = pbar.n
        total_iter = len(train_loader)
        elapsed = time.time() - start_time
        avg_iter_time = elapsed / (current_iter if current_iter else 1)
        rem_time = (total_iter - current_iter) * avg_iter_time

        pbar.set_postfix(loss=f"{cur_loss:.4f}", acc=f"{cur_acc:.4f}", rem_time=f"{rem_time:.2f}s")
    epoch_time = time.time() - start_time
    throughput = total_samples / epoch_time

    # Beregn ekstra metrikker
    precision = precision_score(all_train_labels, all_train_preds, average="macro")
    recall = recall_score(all_train_labels, all_train_preds, average="macro")
    f1 = f1_score(all_train_labels, all_train_preds, average="macro")

    print(
        f"Epoch {epoch} - Loss: {cur_loss:.4f}, Acc: {cur_acc:.4f}, Prec: {precision:.4f}, Rec: {recall:.4f}, F1: {f1:.4f}, Time: {epoch_time:.2f}s, Throughput: {throughput:.2f} samples/s")
    return cur_loss, cur_acc


# Valideringsløkke med progress bar og ekstra metrikker
def validate_epoch(model, val_loader, epoch):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_val_preds = []
    all_val_labels = []
    start_time = time.time()
    pbar = tqdm(val_loader, desc=f"Validation Epoch {epoch}", leave=True)
    with torch.no_grad():
        for batch in pbar:
            batch = batch.to(DEVICE)
            outputs = model(batch)
            targets = batch.y.long()
            loss = F.cross_entropy(outputs, targets)
            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            preds = outputs.argmax(dim=1)
            total_correct += (preds == targets).sum().item()
            total_samples += batch_size

            # Akkumuler for metrikker
            all_val_preds.extend(preds.cpu().numpy())
            all_val_labels.extend(targets.cpu().numpy())

            cur_loss = total_loss / total_samples
            cur_acc = total_correct / total_samples
            current_iter = pbar.n
            total_iter = len(val_loader)
            elapsed = time.time() - start_time
            avg_iter_time = elapsed / (current_iter if current_iter else 1)
            rem_time = (total_iter - current_iter) * avg_iter_time

            pbar.set_postfix(loss=f"{cur_loss:.4f}", acc=f"{cur_acc:.4f}", rem_time=f"{rem_time:.2f}s")
    epoch_time = time.time() - start_time

    # Beregn ekstra metrikker
    precision = precision_score(all_val_labels, all_val_preds, average="macro")
    recall = recall_score(all_val_labels, all_val_preds, average="macro")
    f1 = f1_score(all_val_labels, all_val_preds, average="macro")

    print(
        f"Epoch {epoch} - Val Loss: {cur_loss:.4f}, Val Acc: {cur_acc:.4f}, Prec: {precision:.4f}, Rec: {recall:.4f}, F1: {f1:.4f}, Time: {epoch_time:.2f}s")
    return cur_loss, cur_acc


# Testløkke med progress bar og ekstra metrikker
def test_model(model, test_loader):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    start_time = time.time()
    pbar = tqdm(test_loader, desc="Testing", leave=True)
    with torch.no_grad():
        for batch in pbar:
            batch = batch.to(DEVICE)
            outputs = model(batch)
            targets = batch.y.long()
            loss = F.cross_entropy(outputs, targets)
            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            preds = outputs.argmax(dim=1)
            total_correct += (preds == targets).sum().item()
            total_samples += batch_size
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
    epoch_time = time.time() - start_time
    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples

    # Ekstra metrikker
    precision = precision_score(all_labels, all_preds, average="macro")
    recall = recall_score(all_labels, all_preds, average="macro")
    f1 = f1_score(all_labels, all_preds, average="macro")

    print(
        f"Test - Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}, Prec: {precision:.4f}, Rec: {recall:.4f}, F1: {f1:.4f}, Time: {epoch_time:.2f}s")
    return avg_loss, avg_acc, all_preds, all_labels


if __name__ == "__main__":
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)

    train_dataset = ProcessedDataset(os.path.join(args.output_dir, "train"))
    val_dataset = ProcessedDataset(os.path.join(args.output_dir, "val"))
    test_dataset = ProcessedDataset(os.path.join(args.output_dir, "test"))

    train_loader = DataLoader(train_dataset, num_workers=4, batch_size=args.batch_size, shuffle=True, drop_last=True,
                              persistent_workers=True)
    val_loader = DataLoader(val_dataset, num_workers=4, batch_size=args.batch_size, shuffle=False, drop_last=False,
                            persistent_workers=True)
    test_loader = DataLoader(test_dataset, num_workers=4, batch_size=args.batch_size, shuffle=False, drop_last=False,
                             persistent_workers=True)

    model = ParticleNet({
        "conv_params": args.conv_params,
        "fc_params": args.fc_params,
        "input_features": args.input_features,
        "output_classes": args.output_classes,
    }).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    best_val_loss = float('inf')
    best_epoch = -1
    num_epochs = args.epochs

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, optimizer, train_loader, epoch)
        val_loss, val_acc = validate_epoch(model, val_loader, epoch)

        # Lagre sjekkpunkt hvis val_loss forbedres
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            checkpoint_path = os.path.join(args.output_dir, "checkpoints", f"best_model_epoch_{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
            }, checkpoint_path)
            print(f"Checkpoint lagret ved epoch {epoch} med val_loss: {val_loss:.4f}")

    print("Testing på testsett...")
    test_loss, test_acc, all_preds, all_labels = test_model(model, test_loader)

    # Last beste modell for endelig evaluering
    best_checkpoint_path = os.path.join(args.output_dir, "checkpoints", f"best_model_epoch_{best_epoch}.pt")
    checkpoint = torch.load(best_checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    _, _, all_preds, all_labels = test_model(model, test_loader)

    # Confusion matrix og klassifikasjonsrapport
    conf_matrix = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=list(map(str, range(args.output_classes))),
                yticklabels=list(map(str, range(args.output_classes))))
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(args.output_dir, "confusion_matrix.png"))
    plt.show()

    print(classification_report(all_labels, all_preds, digits=4))

    # Feature importance (gjennomsnittlig absoluttverdi av input_bn.weight)
    feature_importance = torch.mean(torch.abs(model.input_bn.weight)).cpu().numpy()
    print(f"Feature Importance: {feature_importance}")
    with open(os.path.join(args.output_dir, "feature_importance.txt"), "w") as f:
        f.write(f"Feature Importance: {feature_importance}\n")

    print("✅ Treningsprosess fullført!")

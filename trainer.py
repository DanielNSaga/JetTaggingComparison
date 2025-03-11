import time
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
import matplotlib.pyplot as plt
import psutil
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, classification_report
from particlenet import ParticleNet
from args import Args
from tqdm import tqdm
from torch_geometric.data import Batch, Data

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    scaler = torch.cuda.amp.GradScaler()

args = Args()

class StreamingDataset(IterableDataset):
    def __init__(self, dataset_path, indices):
        super().__init__()
        self.dataset_path = dataset_path
        self.indices = indices

    def __iter__(self):
        data, slices = torch.load(self.dataset_path, weights_only=False)
        for idx in self.indices:
            start_idx = slices["x"][idx]
            end_idx   = slices["x"][idx+1]
            yield Data(
                x=data.x[start_idx:end_idx],
                y=data.y[idx],
                pos=data.pos[start_idx:end_idx]
            )

def collate_fn(batch):
    return Batch.from_data_list(batch)

def fpr_at_90_recall(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    idx = np.where(tpr >= 0.90)[0][0]
    return fpr[idx] if idx < len(fpr) else fpr[-1]

@torch.no_grad()
def measure_inference_latency(model, loader, num_samples=1000):
    model.eval()
    times = []
    for count, batch in enumerate(loader):
        if count >= num_samples:
            break
        batch = batch.to(DEVICE)
        with torch.cuda.amp.autocast():
            start = time.time()
            _ = model(batch)
            end   = time.time()
        times.append(end - start)
    if len(times) == 0:
        return 0.0
    return np.mean(times) * 1000.0

def train_epoch(model, optimizer, train_loader, epoch):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    pbar = tqdm(
        train_loader,
        desc=f"Epoch {epoch}",
        leave=True,
        total=TRAIN_BATCHES_TOTAL
    )

    for i, batch in enumerate(pbar):
        batch = batch.to(DEVICE)
        batch.x = batch.x.to(torch.float16)  # ✅ Kun x til FP16
        batch.pos = batch.pos.to(torch.float16)  # ✅ Konverter pos også om nødvendig

        optimizer.zero_grad()

        with torch.amp.autocast(device_type="cuda"):
            outputs = model(batch)
            targets = batch.y.long()  # ✅ Sørg for at labels forblir int64
            loss = F.cross_entropy(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = targets.size(0)
        total_loss += loss.item() * bs
        total_correct += (outputs.argmax(dim=1) == targets).sum().item()
        total_samples += bs

        cur_loss = total_loss / total_samples
        cur_acc  = total_correct / total_samples
        pbar.set_postfix(loss=f"{cur_loss:.4f}", acc=f"{cur_acc:.4f}")

        if i + 1 >= TRAIN_BATCHES_TOTAL:
            break

    return total_loss / total_samples, total_correct / total_samples


def validate_epoch(model, loader, epoch_str="Val", is_val=True):
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    all_preds, all_labels, all_probs = [], [], []

    # Hardkodet total for tqdm
    total_steps = VAL_BATCHES_TOTAL if is_val else TEST_BATCHES_TOTAL
    pbar = tqdm(loader, desc=f"{epoch_str}", leave=True, total=total_steps)

    with torch.no_grad():
        for i, batch in enumerate(pbar):
            batch = batch.to(DEVICE)
            with torch.cuda.amp.autocast():
                out = model(batch)
            targets = batch.y.long()
            loss = F.cross_entropy(out, targets)

            preds = out.argmax(dim=1)
            probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()

            bs = targets.size(0)
            total_loss    += loss.item() * bs
            total_correct += (preds == targets).sum().item()
            total_samples += bs

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
            all_probs.extend(probs)

            cur_loss = total_loss / total_samples
            cur_acc  = total_correct / total_samples
            pbar.set_postfix(loss=f"{cur_loss:.4f}", acc=f"{cur_acc:.4f}")

            if i+1 >= total_steps:
                break

    roc_auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels))>1 else float('nan')
    fpr90   = fpr_at_90_recall(all_labels, all_probs)
    return total_loss / total_samples, total_correct / total_samples, roc_auc, fpr90

@torch.no_grad()
def permutation_feature_importance(model, loader, input_dim, n_perm=3):
    model.eval()
    baseline_acc = validate_epoch(model, loader, epoch_str="Baseline", is_val=False)[1]
    importances = np.zeros(input_dim, dtype=np.float32)

    for feat_idx in range(input_dim):
        results = []
        for _ in range(n_perm):
            for batch in loader:
                batch = batch.to(DEVICE)
                batch.x = batch.x.to(torch.float16)  # ✅ Kun x til FP16
                batch.pos = batch.pos.to(torch.float16)  # ✅ Hvis nødvendig

                original_vals = batch.x[:, feat_idx].clone()
                permuted_vals = original_vals[torch.randperm(len(original_vals))]
                batch.x[:, feat_idx] = permuted_vals

                perm_acc = validate_epoch(model, [batch], epoch_str="PermTest", is_val=False)[1]
                results.append(baseline_acc - perm_acc)
                batch.x[:, feat_idx] = original_vals  # Reset

        importances[feat_idx] = float(np.mean(results))
        print(f"Feature {feat_idx:2d} => importance={importances[feat_idx]:.4f}")

    return importances



if __name__ == "__main__":

    # 🔹 **Optimaliseringer for A100**
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        scaler = torch.cuda.amp.GradScaler()

    args = Args()

    # 🔹 **Lese metadata fra `data.pt`**
    print("📥 Leser `data.pt` for å hente metadata...")
    data, slices = torch.load("data.pt", weights_only=False)
    n_total = len(data.y)
    print(f"📊 Dataset har {n_total} jets og {data.x.shape[0]} partikler.")

    # 🔹 **Splitt dataset (80-10-10)**
    torch.manual_seed(42)
    idx = torch.randperm(n_total)

    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    n_test = n_total - n_train - n_val

    train_loader = DataLoader(StreamingDataset("data.pt", idx[:n_train]), batch_size=args.batch_size, num_workers=0, collate_fn=collate_fn, pin_memory=True)
    val_loader   = DataLoader(StreamingDataset("data.pt", idx[n_train:n_train + n_val]), batch_size=args.batch_size, num_workers=8, collate_fn=collate_fn, pin_memory=True)
    test_loader  = DataLoader(StreamingDataset("data.pt", idx[n_train + n_val:]), batch_size=args.batch_size, num_workers=8, collate_fn=collate_fn, pin_memory=True)

    # 🔹 **Opprett ParticleNet-modell**
    print("🚀 Oppretter ParticleNet-modell...")
    model = ParticleNet({
        "input_features": args.input_features,
        "output_classes": args.output_classes,
        "conv_params": args.conv_params,
        "fc_params": args.fc_params,
    }).to(DEVICE)

    model = torch.compile(model)  # ✅ Optimaliser for A100
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, fused=True)  # ✅ Fused AdamW

    # 🔹 **Sett faste antall batches**
    TRAIN_BATCHES_TOTAL = 12500
    VAL_BATCHES_TOTAL   = 1562
    TEST_BATCHES_TOTAL  = 1562

    best_val_loss = float('inf')
    best_epoch    = -1

    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, optimizer, train_loader, epoch)
        val_loss, val_acc, val_roc, val_fpr90 = validate_epoch(model, val_loader, epoch_str=f"Val epoch={epoch}", is_val=True)

        # **Lagre best modell basert på valideringstap**
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save({"model_state_dict": model.state_dict()}, "best_model.pt")
            print(f"🔹 Lagret checkpoint ved epoch {epoch}")

    # 🔹 **Last beste modell**
    print(f"📥 Laster beste checkpoint fra epoch {best_epoch}")
    model.load_state_dict(torch.load("best_model.pt")["model_state_dict"])
    model.to(DEVICE)

    # 🔹 **Endelig test på test-settet**
    print("📊 **Endelig evaluering på test-sett**")
    final_loss, final_acc, final_roc, final_fpr90 = validate_epoch(model, test_loader, epoch_str="FinalTest", is_val=False)
    print(f"Test => Loss={final_loss:.4f}, Acc={final_acc:.4f}, ROC={final_roc:.4f}, FPR90={final_fpr90:.4f}")

    # 🔹 **Inference Latency**
    latency = measure_inference_latency(model, test_loader)
    print(f"🕒 Inference-latency per batch ~ {latency:.3f} ms")

    # 🔹 **Memory Usage**
    memory_usage = psutil.virtual_memory().used / 1e9
    print(f"💾 Memory usage ~ {memory_usage:.2f} GB")

    # 🔹 **Confusion Matrix**
    print("📊 Genererer Confusion Matrix...")
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(DEVICE)
            with torch.cuda.amp.autocast():
                out = model(batch)
            preds = out.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.show()

    # 🔹 **Klassifikasjonsrapport**
    print("📊 Klassifikasjonsrapport:")
    print(classification_report(all_labels, all_preds, digits=4))

    # 🔹 **Permutasjonsbasert Feature Importance**
    print("📊 **Kjører Permutasjonsbasert Feature Importance**")
    importances = permutation_feature_importance(
        model, test_loader,
        input_dim=args.input_features,
        n_perm=3
    )
    print("Permutation-based importances:\n", importances)

    # 🔹 **Lagre og plotte feature importance**
    np.savetxt("permutation_importances.txt", importances, fmt="%.4f")
    plt.figure(figsize=(7, 4))
    plt.bar(range(len(importances)), importances)
    plt.xlabel("Feature index")
    plt.ylabel("Importance (Δ acc)")
    plt.title("Permutation-based Feature Importances")
    plt.savefig("permutation_importances.png")
    plt.show()

    print("✅ **Fullført!**")

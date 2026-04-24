from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from stgcn_model import STGCNClassifier


def load_stgcn_npz(npz_path: Path):
    """
    Load prepared ST-GCN windows from NPZ.
    """
    if not npz_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    x = data["X"].astype(np.float32)       # [N, C, T, V, M]
    y = data["y"].astype(np.int64)         # [N]
    classes = data["classes"].tolist()     # list[str]
    vid_ids = data["vid_ids"] if "vid_ids" in data.files else np.arange(len(y))

    print(f"Loaded: {npz_path}")
    print(f"X shape: {x.shape}")
    print(f"y shape: {y.shape}")
    print(f"vid_ids shape: {vid_ids.shape}")
    print(f"Classes: {classes}")
    return x, y, classes, vid_ids


def maybe_subsample_by_videos(
    x,
    y,
    vid_ids,
    max_videos: int | None,
    random_state: int = 42,
):
    """
    Optionally reduce dataset by selecting a subset of videos.
    """
    unique_vids = np.unique(vid_ids)
    if max_videos is None or max_videos <= 0 or max_videos >= len(unique_vids):
        return x, y, vid_ids

    # Derive one class label per video for stratified video-level sampling.
    video_labels = {}
    for vid in unique_vids:
        idx = np.where(vid_ids == vid)[0][0]
        video_labels[vid] = y[idx]
    y_vid = np.array([video_labels[v] for v in unique_vids], dtype=np.int64)

    try:
        vids_sub, _ = train_test_split(
            unique_vids,
            train_size=max_videos,
            stratify=y_vid,
            random_state=random_state,
        )
    except ValueError:
        # Fallback when stratification is not possible with small subsets.
        rng = np.random.default_rng(random_state)
        vids_sub = rng.choice(unique_vids, size=max_videos, replace=False)

    mask = np.isin(vid_ids, vids_sub)
    x_sub, y_sub, vid_sub = x[mask], y[mask], vid_ids[mask]
    print(f"Using subsampled videos: {len(np.unique(vid_sub))} videos, {len(y_sub)} windows")
    return x_sub, y_sub, vid_sub


def build_dataloaders(
    x,
    y,
    vid_ids,
    batch_size: int,
    random_state: int = 42,
    num_workers: int = 0,
    pin_memory: bool = False,
):
    """
    Build train/val loaders with video-level split to prevent leakage.
    """
    unique_vids = np.unique(vid_ids)

    # Derive one class label per video for stratified video-level split.
    video_labels = {}
    for vid in unique_vids:
        idx = np.where(vid_ids == vid)[0][0]
        video_labels[vid] = y[idx]
    y_vid = np.array([video_labels[v] for v in unique_vids], dtype=np.int64)

    try:
        train_vids, val_vids = train_test_split(
            unique_vids,
            test_size=0.2,
            stratify=y_vid,
            random_state=random_state,
        )
    except ValueError:
        # Fallback when stratification is not feasible.
        train_vids, val_vids = train_test_split(
            unique_vids,
            test_size=0.2,
            random_state=random_state,
        )

    train_mask = np.isin(vid_ids, train_vids)
    val_mask = np.isin(vid_ids, val_vids)

    x_train, y_train = x[train_mask], y[train_mask]
    x_val, y_val = x[val_mask], y[val_mask]

    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=bool(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=bool(num_workers > 0),
    )

    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples: {len(val_ds)}")
    print(f"Train videos: {len(np.unique(train_vids))}")
    print(f"Val videos: {len(np.unique(val_vids))}")
    return train_loader, val_loader, y_val


def run_epoch(model, loader, criterion, optimizer, device, scaler):
    """
    Run one training epoch.
    """
    model.train()
    total_loss = 0.0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = model(xb)
            loss = criterion(logits, yb)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * xb.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device, criterion: nn.Module | None = None):
    """
    Evaluate model: accuracy, predictions, and optional mean cross-entropy loss.
    """
    model.eval()
    preds = []
    trues = []
    total_loss = 0.0
    n_samples = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        if criterion is not None:
            total_loss += criterion(logits, yb).item() * xb.size(0)
            n_samples += xb.size(0)
        pred = torch.argmax(logits, dim=1).cpu().numpy()
        preds.append(pred)
        trues.append(yb.cpu().numpy())

    y_pred = np.concatenate(preds)
    y_true = np.concatenate(trues)
    acc = accuracy_score(y_true, y_pred)
    mean_loss = (total_loss / n_samples) if criterion is not None and n_samples > 0 else float("nan")
    return acc, y_true, y_pred, mean_loss


def save_confusion(y_true, y_pred, classes, out_path: Path, title: str = "ST-GCN confusion matrix"):
    """
    Save confusion matrix plot.
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap="Blues")
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    print(f"Confusion matrix saved to: {out_path}")


def save_training_curves(
    epochs: list[int],
    train_losses: list[float],
    val_losses: list[float],
    val_accs: list[float],
    out_path: Path,
) -> None:
    """
    Save train/validation loss and validation accuracy on one figure.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 4))
    ax_loss.plot(epochs, train_losses, label="Train loss", marker="o", markersize=3)
    ax_loss.plot(epochs, val_losses, label="Val loss", marker="o", markersize=3)
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Cross-entropy")
    ax_loss.set_title("ST-GCN loss")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    ax_acc.plot(epochs, val_accs, color="tab:green", label="Val accuracy", marker="o", markersize=3)
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("ST-GCN validation accuracy")
    ax_acc.set_ylim(0.0, 1.0)
    ax_acc.legend()
    ax_acc.grid(True, alpha=0.3)

    fig.suptitle("ST-GCN training curves")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Training curves saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Train ST-GCN on prepared skeleton windows.")
    parser.add_argument("--data", type=str, default="data/stgcn/stgcn_windows.npz")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay (try 2e-4–5e-4 if overfitting).")
    parser.add_argument(
        "--best-metric",
        type=str,
        default="val_loss",
        choices=("val_loss", "val_acc"),
        help="Checkpoint + early stopping: val_loss (lower) usually tracks noisy val better than val_acc alone.",
    )
    parser.add_argument(
        "--no-lr-scheduler",
        action="store_true",
        help="Disable ReduceLROnPlateau (default: scheduler on, steps on val loss).",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=0)
    parser.add_argument("--fast", action="store_true", help="Quick mode for faster experiments")
    args = parser.parse_args()

    if args.fast:
        args.epochs = min(args.epochs, 15)
        args.batch_size = max(args.batch_size, 128)
        args.patience = min(args.patience, 4)
        if args.max_videos <= 0:
            args.max_videos = 240

    if args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)

    npz_path = Path(args.data)
    x, y, classes, vid_ids = load_stgcn_npz(npz_path)
    x, y, vid_ids = maybe_subsample_by_videos(
        x, y, vid_ids, max_videos=args.max_videos, random_state=42
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
    else:
        print("CUDA not available. Use CUDA-enabled PyTorch to leverage your RTX 4060 Ti.")

    train_loader, val_loader, _ = build_dataloaders(
        x,
        y,
        vid_ids,
        batch_size=args.batch_size,
        num_workers=max(args.num_workers, 0),
        pin_memory=(device.type == "cuda"),
    )

    model = STGCNClassifier(num_classes=len(classes), base_channels=args.base_channels).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))
    scheduler = None if args.no_lr_scheduler else ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )

    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_state = None
    best_true = None
    best_pred = None
    no_improve = 0

    epoch_list: list[int] = []
    train_losses: list[float] = []
    val_losses: list[float] = []
    val_accs: list[float] = []

    print("\nTraining ST-GCN...\n")
    print(
        f"Optimiser: AdamW lr={args.lr} weight_decay={args.weight_decay} | "
        f"best_metric={args.best_metric} | "
        f"lr_scheduler={'off' if scheduler is None else 'ReduceLROnPlateau(val_loss)'}"
    )
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_acc, y_true, y_pred, val_loss = evaluate(model, val_loader, device, criterion)
        if scheduler is not None:
            scheduler.step(val_loss)
        lr_now = optimizer.param_groups[0]["lr"]
        epoch_list.append(epoch)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        print(
            f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f} | lr={lr_now:.2e}"
        )

        if args.best_metric == "val_acc":
            improved = val_acc > best_val_acc
        else:
            improved = val_loss < best_val_loss

        if improved:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            best_true = y_true
            best_pred = y_pred
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch:02d} (no improvement for {args.patience} epochs).")
                break

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "stgcn_best.pth"
    torch.save(best_state, model_path)
    print(
        f"\nBest checkpoint ({args.best_metric}): val_acc={best_val_acc:.4f} val_loss={best_val_loss:.4f}"
    )
    print(f"Model saved to: {model_path}")

    label_info = {"classes": classes, "base_channels": args.base_channels}
    label_path = models_dir / "stgcn_label_info.pkl"
    joblib.dump(label_info, label_path)
    print(f"Label info saved to: {label_path}")

    docs_dir = Path("docs")
    save_training_curves(
        epoch_list,
        train_losses,
        val_losses,
        val_accs,
        docs_dir / "stgcn_training_curves.png",
    )

    if best_state is not None:
        model.load_state_dict(best_state)

    save_confusion(
        best_true,
        best_pred,
        classes,
        docs_dir / "confusion_matrix_stgcn_val.png",
        title="ST-GCN validation (best checkpoint)",
    )
    save_confusion(
        best_true,
        best_pred,
        classes,
        docs_dir / "confusion_matrix_stgcn.png",
        title="ST-GCN validation (best checkpoint)",
    )

    _, y_train_true, y_train_pred, _ = evaluate(model, train_loader, device, None)
    save_confusion(
        y_train_true,
        y_train_pred,
        classes,
        docs_dir / "confusion_matrix_stgcn_train.png",
        title="ST-GCN training (best checkpoint)",
    )


if __name__ == "__main__":
    main()

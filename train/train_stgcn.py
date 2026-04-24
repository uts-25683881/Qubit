"""
Train ST-GCN on NPZ windows: video-level train/val split, optional LR schedule,
early stopping, then save weights, plots, and confusion matrices under docs/.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from stgcn_model import STGCNClassifier

# Fraction of videos (not windows) held out for validation.
VAL_FRACTION = 0.2


def _video_ids_and_labels(vid_ids: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One label per video (from the first window of that video)."""
    unique = np.unique(vid_ids)
    per_vid = np.array([int(y[np.where(vid_ids == v)[0][0]]) for v in unique], dtype=np.int64)
    return unique, per_vid


def load_stgcn_npz(npz_path: Path) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Load X [N,C,T,V,M], y, class names, and per-window video ids from NPZ."""
    if not npz_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    x = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    classes = data["classes"].tolist()
    # Fallback ids: one pseudo-video per window (no real vid_id in file).
    vids = data["vid_ids"] if "vid_ids" in data.files else np.arange(len(y))

    print(f"Loaded: {npz_path}\nX {x.shape} | y {y.shape} | vids {vids.shape}\nClasses: {classes}")
    return x, y, classes, vids


def build_dataloaders(
    x: np.ndarray,
    y: np.ndarray,
    vid_ids: np.ndarray,
    batch_size: int,
    random_state: int = 42,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """Split by video id, then assign all windows of each video to train or val only."""
    unique, y_vid = _video_ids_and_labels(vid_ids, y)
    try:
        train_v, val_v = train_test_split(
            unique, test_size=VAL_FRACTION, stratify=y_vid, random_state=random_state
        )
    except ValueError:
        # Too few samples per class for stratify — fall back to random split.
        train_v, val_v = train_test_split(unique, test_size=VAL_FRACTION, random_state=random_state)

    train_m, val_m = np.isin(vid_ids, train_v), np.isin(vid_ids, val_v)
    x_tr, y_tr = x[train_m], y[train_m]
    x_va, y_va = x[val_m], y[val_m]

    dl_kw: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": bool(num_workers > 0),
    }
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr)), shuffle=True, **dl_kw
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_va), torch.from_numpy(y_va)), shuffle=False, **dl_kw
    )

    print(f"Train: {len(train_loader.dataset)} windows / {np.unique(train_v).size} videos")
    print(f"Val:   {len(val_loader.dataset)} windows / {np.unique(val_v).size} videos")
    return train_loader, val_loader


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler,
) -> float:
    """One pass over train_loader; returns mean cross-entropy (AMP on CUDA)."""
    model.train()
    total = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            loss = criterion(model(xb), yb)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += loss.item() * xb.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module | None = None,
) -> tuple[float, np.ndarray, np.ndarray, float]:
    """Returns (accuracy, y_true, y_pred, mean_loss or nan if criterion is None)."""
    model.eval()
    preds, trues = [], []
    loss_sum = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        if criterion is not None:
            loss_sum += criterion(logits, yb).item() * xb.size(0)
        preds.append(torch.argmax(logits, dim=1).cpu().numpy())
        trues.append(yb.cpu().numpy())

    y_pred = np.concatenate(preds)
    y_true = np.concatenate(trues)
    acc = accuracy_score(y_true, y_pred)
    n_ds = len(loader.dataset)
    mean_loss = (loss_sum / n_ds) if criterion is not None and n_ds else float("nan")
    return acc, y_true, y_pred, mean_loss


def save_confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: list[str],
    out_path: Path,
    title: str = "ST-GCN confusion matrix",
) -> None:
    """Plot and save a sklearn confusion matrix PNG."""
    disp = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(y_true, y_pred), display_labels=classes
    )
    disp.plot(cmap="Blues")
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    print(f"Confusion matrix -> {out_path}")


def save_training_curves(
    epochs: list[int],
    train_losses: list[float],
    val_losses: list[float],
    val_accs: list[float],
    out_path: Path,
) -> None:
    """Loss (train vs val) and val accuracy vs epoch — one figure."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_l, ax_a) = plt.subplots(1, 2, figsize=(10, 4))

    ax_l.plot(epochs, train_losses, "o-", ms=3, label="Train loss")
    ax_l.plot(epochs, val_losses, "o-", ms=3, label="Val loss")
    ax_l.set(xlabel="Epoch", ylabel="Cross-entropy", title="ST-GCN loss")
    ax_l.legend()
    ax_l.grid(True, alpha=0.3)

    ax_a.plot(epochs, val_accs, "o-", ms=3, color="tab:green", label="Val accuracy")
    ax_a.set(xlabel="Epoch", ylabel="Accuracy", title="ST-GCN validation accuracy", ylim=(0, 1))
    ax_a.legend()
    ax_a.grid(True, alpha=0.3)

    fig.suptitle("ST-GCN training curves")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Training curves -> {out_path}")


def _is_better(metric: str, val_acc: float, val_loss: float, best_acc: float, best_loss: float) -> bool:
    """Whether this epoch beats the best so far (higher acc or lower loss)."""
    if metric == "val_acc":
        return val_acc > best_acc
    return val_loss < best_loss


def run_training_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    scheduler: ReduceLROnPlateau | None,
    device: torch.device,
    epochs: int,
    patience: int,
    best_metric: str,
) -> dict[str, Any]:
    """Train for up to `epochs` epochs; early-stop if `best_metric` stalls `patience` times."""
    best_acc, best_loss = 0.0, float("inf")
    best_state = best_true = best_pred = None
    no_improve = 0

    ep, tr_l, va_l, va_a = [], [], [], []

    for epoch in range(1, epochs + 1):
        tr_loss = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        acc, yt, yp, va_loss = evaluate(model, val_loader, device, criterion)
        if scheduler is not None:
            scheduler.step(va_loss)

        lr = optimizer.param_groups[0]["lr"]
        ep.append(epoch)
        tr_l.append(tr_loss)
        va_l.append(va_loss)
        va_a.append(acc)
        print(f"Epoch {epoch:02d} | train_loss={tr_loss:.4f} | val_loss={va_loss:.4f} | val_acc={acc:.4f} | lr={lr:.2e}")

        if _is_better(best_metric, acc, va_loss, best_acc, best_loss):
            best_acc, best_loss = acc, va_loss
            # CPU copy for torch.save (portable across devices).
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            best_true, best_pred = yt, yp
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch:02d} (no {best_metric} improvement for {patience} epochs).")
                break

    return {
        "epoch_list": ep,
        "train_losses": tr_l,
        "val_losses": va_l,
        "val_accs": va_a,
        "best_state": best_state,
        "best_true": best_true,
        "best_pred": best_pred,
        "best_val_acc": best_acc,
        "best_val_loss": best_loss,
    }


def parse_args() -> argparse.Namespace:
    """CLI: data path, optimisation, DataLoader, checkpoint metric."""
    p = argparse.ArgumentParser(description="Train ST-GCN on prepared skeleton windows.")
    p.add_argument("--data", type=str, default="data/stgcn/stgcn_windows.npz")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW L2 penalty.")
    p.add_argument(
        "--best-metric",
        choices=("val_loss", "val_acc"),
        default="val_loss",
        help="Metric for checkpointing and early stopping.",
    )
    p.add_argument("--no-lr-scheduler", action="store_true", help="Disable ReduceLROnPlateau on val loss.")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--cpu-threads", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)

    # --- Data ---
    x, y, classes, vid_ids = load_stgcn_npz(Path(args.data))

    # --- Device & loaders ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"CUDA: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True  # Faster fixed-size convs on GPU.

    train_loader, val_loader = build_dataloaders(
        x,
        y,
        vid_ids,
        batch_size=args.batch_size,
        num_workers=max(args.num_workers, 0),
        pin_memory=device.type == "cuda",
    )

    # --- Model & optimisation ---
    model = STGCNClassifier(num_classes=len(classes), base_channels=args.base_channels).to(device)
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    scheduler = (
        None
        if args.no_lr_scheduler
        else ReduceLROnPlateau(optimiser, mode="min", factor=0.5, patience=2, min_lr=1e-6)
    )

    print("\nTraining ST-GCN")
    print(
        f"AdamW lr={args.lr} wd={args.weight_decay} | best_metric={args.best_metric} | "
        f"scheduler={'off' if scheduler is None else 'ReduceLROnPlateau(val_loss)'}\n"
    )

    out = run_training_loop(
        model,
        train_loader,
        val_loader,
        criterion,
        optimiser,
        scaler,
        scheduler,
        device,
        epochs=args.epochs,
        patience=args.patience,
        best_metric=args.best_metric,
    )

    # --- Save weights & label metadata ---
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    torch.save(out["best_state"], models_dir / "stgcn_best.pth")
    print(f"\nBest ({args.best_metric}): val_acc={out['best_val_acc']:.4f} val_loss={out['best_val_loss']:.4f}")
    print(f"Model -> {models_dir / 'stgcn_best.pth'}")

    joblib.dump(
        {"classes": classes, "base_channels": args.base_channels},
        models_dir / "stgcn_label_info.pkl",
    )
    print(f"Labels -> {models_dir / 'stgcn_label_info.pkl'}")

    # --- Reports (curves + confusion matrices) ---
    docs = Path("docs")
    save_training_curves(
        out["epoch_list"],
        out["train_losses"],
        out["val_losses"],
        out["val_accs"],
        docs / "stgcn_training_curves.png",
    )

    if out["best_state"] is not None:
        model.load_state_dict(out["best_state"])

    # Val confusion: two filenames (legacy + explicit val name).
    for name, title in (
        ("confusion_matrix_stgcn_val.png", "ST-GCN validation (best checkpoint)"),
        ("confusion_matrix_stgcn.png", "ST-GCN validation (best checkpoint)"),
    ):
        save_confusion(out["best_true"], out["best_pred"], classes, docs / name, title=title)

    # Train confusion at same best weights (diagnostic only).
    _, yt_tr, yp_tr, _ = evaluate(model, train_loader, device, None)
    save_confusion(yt_tr, yp_tr, classes, docs / "confusion_matrix_stgcn_train.png", "ST-GCN training (best checkpoint)")


if __name__ == "__main__":
    main()

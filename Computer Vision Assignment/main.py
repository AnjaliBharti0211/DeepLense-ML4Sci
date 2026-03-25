"""
Gravitational Lensing Dark Matter Classification Pipeline
==========================================================
A Kaggle-ready PyTorch pipeline for classifying gravitational lensing images
into three categories: no_substructure (0), subhalo (1), vortex (2).

Based on:
  - Varma et al. (2020): https://arxiv.org/pdf/2005.05353
  - Alexander et al. (2020): https://doi.org/10.3847/1538-4357/ab7925

Usage:
  Kaggle:  Run as a notebook cell (auto-detects /kaggle/input/).
  Local:   python main.py --data_dir ./data --dry-run
"""

import argparse
import os
import sys
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────────────
# 1. Configuration
# ──────────────────────────────────────────────────────────────────────────────

def get_config(args=None):
    """Return a namespace with all hyper-parameters and paths."""
    parser = argparse.ArgumentParser(description="Gravitational Lensing Classifier")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Root directory containing the 3 class folders. "
                             "Defaults to /kaggle/working/dataset on Kaggle.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Where to save the model & plots. "
                             "Defaults to /kaggle/working/ or ./output.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, default=224,
                        help="Resize images to this square size.")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true",
                        help="Run a single forward/backward pass on dummy data "
                             "to verify the pipeline, then exit.")
    cfg = parser.parse_args(args if args is not None else sys.argv[1:])

    # ---------- resolve paths ----------
    on_kaggle = Path("/kaggle/input").exists()

    if cfg.data_dir is None:
        cfg.data_dir = "/kaggle/working/dataset" if on_kaggle else "./data"

    if cfg.output_dir is None:
        cfg.output_dir = "/kaggle/working" if on_kaggle else "./output"

    os.makedirs(cfg.output_dir, exist_ok=True)
    cfg.on_kaggle = on_kaggle
    return cfg


# ──────────────────────────────────────────────────────────────────────────────
# 2. Dataset Extraction (Kaggle)
# ──────────────────────────────────────────────────────────────────────────────

def extract_dataset(cfg):
    """Extract dataset.zip on Kaggle if not already extracted."""
    if not cfg.on_kaggle:
        return  # Nothing to extract locally

    zip_path = "/kaggle/input/dataset.zip"
    # Also check for common Kaggle dataset naming patterns
    if not os.path.exists(zip_path):
        input_dir = Path("/kaggle/input")
        zips = list(input_dir.rglob("*.zip"))
        if zips:
            zip_path = str(zips[0])
        else:
            print("[INFO] No zip file found — assuming data is already extracted.")
            return

    target = Path(cfg.data_dir)
    if target.exists() and any(target.iterdir()):
        print(f"[INFO] Data already extracted at {target}")
        return

    print(f"[INFO] Extracting {zip_path} → {target} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target.parent)
    print("[INFO] Extraction complete.")


# ──────────────────────────────────────────────────────────────────────────────
# 3. LensingDataset
# ──────────────────────────────────────────────────────────────────────────────

CLASS_NAMES = ["no_substructure", "subhalo", "vortex"]
CLASS_TO_LABEL = {name: idx for idx, name in enumerate(CLASS_NAMES)}


class LensingDataset(Dataset):
    """
    Expects the following directory layout:
        root/
          no_substructure/  *.png | *.npy
          subhalo/          *.png | *.npy
          vortex/           *.png | *.npy

    Images are min-max normalised — no ImageNet mean/std is applied so that
    subtle lensing signals are preserved (per Alexander et al.).
    """

    SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".npy", ".tif", ".tiff", ".bmp"}

    def __init__(self, root: str, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []

        for cls_name, label in CLASS_TO_LABEL.items():
            cls_dir = self.root / cls_name
            if not cls_dir.is_dir():
                raise FileNotFoundError(
                    f"Expected class directory not found: {cls_dir}"
                )
            for fpath in sorted(cls_dir.iterdir()):
                if fpath.suffix.lower() in self.SUPPORTED_EXT:
                    self.samples.append((fpath, label))

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found under {self.root}")

        print(f"[LensingDataset] Loaded {len(self.samples)} images from {self.root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        # Load image — support both .npy arrays and standard image formats
        if path.suffix.lower() == ".npy":
            img = np.load(str(path)).astype(np.float32)
        else:
            img = np.array(Image.open(path).convert("L"), dtype=np.float32)

        # Min-max normalise to [0, 1]
        img_min, img_max = img.min(), img.max()
        if img_max - img_min > 0:
            img = (img - img_min) / (img_max - img_min)

        # Convert single-channel → 3-channel (ResNet expects 3 channels)
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=0)  # (3, H, W)
        elif img.ndim == 3 and img.shape[-1] in (1, 3):
            img = np.transpose(img, (2, 0, 1))       # (C, H, W)
            if img.shape[0] == 1:
                img = np.repeat(img, 3, axis=0)

        img = torch.from_numpy(img)  # (3, H, W) float32

        if self.transform:
            img = self.transform(img)

        return img, label


# ──────────────────────────────────────────────────────────────────────────────
# 4. Transforms (Symmetrical Augmentation — Alexander et al.)
# ──────────────────────────────────────────────────────────────────────────────

def get_transforms(cfg, train=True):
    """
    Symmetrical augmentation: 90° rotations, horizontal & vertical flips.
    No ImageNet normalisation — only resize + min-max (handled in dataset).
    """
    t = [transforms.Resize((cfg.img_size, cfg.img_size))]
    if train:
        t += [
            transforms.RandomChoice([
                transforms.RandomRotation((0, 0)),
                transforms.RandomRotation((90, 90)),
                transforms.RandomRotation((180, 180)),
                transforms.RandomRotation((270, 270)),
            ]),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
        ]
    return transforms.Compose(t)


# ──────────────────────────────────────────────────────────────────────────────
# 5. Model Construction
# ──────────────────────────────────────────────────────────────────────────────

def build_model(num_classes=3):
    """
    ResNet-18 trained from scratch (weights=None).
    Final FC layer replaced with Linear(512, num_classes).
    Wrapped in DataParallel if ≥ 2 GPUs are available.
    """
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        print(f"[INFO] Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    model = model.to(device)
    return model, device


# ──────────────────────────────────────────────────────────────────────────────
# 6. Training Loop
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    """Train for one epoch with mixed-precision."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="  train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        with autocast(device_type="cuda", enabled=(device.type == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validate and return loss, accuracy, all logits, and all labels."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_logits = []
    all_labels = []

    for images, labels in tqdm(loader, desc="  val  ", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(device_type="cuda", enabled=(device.type == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

        all_logits.append(outputs.float().cpu())
        all_labels.append(labels.cpu())

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    return epoch_loss, epoch_acc, all_logits, all_labels


def train(model, train_loader, val_loader, cfg, device):
    """Full training loop for `cfg.epochs` epochs."""
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )
    scaler = GradScaler(enabled=(device.type == "cuda"))

    best_val_loss = float("inf")
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, cfg.epochs + 1):
        print(f"\nEpoch {epoch}/{cfg.epochs}  "
              f"(lr={optimizer.param_groups[0]['lr']:.2e})")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        val_loss, val_acc, val_logits, val_labels = validate(
            model, val_loader, criterion, device
        )

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"  Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f}  Acc: {val_acc:.4f}")

        # Checkpoint best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(cfg.output_dir, "model.pth")
            state = (model.module.state_dict()
                     if isinstance(model, nn.DataParallel)
                     else model.state_dict())
            torch.save(state, ckpt_path)
            print(f"  ✓ Best model saved → {ckpt_path}")

    return history, val_logits, val_labels


# ──────────────────────────────────────────────────────────────────────────────
# 7. ROC-AUC Evaluation & Visualization
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_roc_auc(logits: torch.Tensor, labels: torch.Tensor, cfg):
    """
    Compute macro One-vs-Rest ROC-AUC and plot per-class ROC curves.

    Args:
        logits: (N, 3) raw model outputs from the final validation pass.
        labels: (N,) ground-truth class indices.
        cfg:    Configuration namespace (for output_dir).
    """
    probs = torch.softmax(logits, dim=1).numpy()
    labels_np = labels.numpy()
    labels_bin = label_binarize(labels_np, classes=[0, 1, 2])

    # Macro ROC-AUC
    macro_auc = roc_auc_score(
        labels_bin, probs, multi_class="ovr", average="macro"
    )
    print(f"\n{'='*50}")
    print(f"  Macro ROC-AUC (OvR): {macro_auc:.4f}")
    print(f"{'='*50}")

    # Per-class ROC curves
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#2196F3", "#FF9800", "#4CAF50"]

    for i, (cls_name, color) in enumerate(zip(CLASS_NAMES, colors)):
        fpr, tpr, _ = roc_curve(labels_bin[:, i], probs[:, i])
        class_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{cls_name} (AUC = {class_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Curves — Macro AUC = {macro_auc:.4f}", fontsize=14)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    roc_path = os.path.join(cfg.output_dir, "roc_curve.png")
    fig.savefig(roc_path, dpi=150)
    plt.close(fig)
    print(f"  ROC curve saved → {roc_path}")

    return macro_auc


# ──────────────────────────────────────────────────────────────────────────────
# 8. Dry-Run (Local Verification)
# ──────────────────────────────────────────────────────────────────────────────

def dry_run():
    """Verify the full pipeline with random data (no real dataset needed)."""
    print("=" * 60)
    print("  DRY RUN — verifying pipeline with dummy data")
    print("=" * 60)

    model, device = build_model(num_classes=3)
    print(f"[✓] Model built on {device}")

    # Dummy forward + backward
    dummy = torch.randn(4, 3, 224, 224, device=device)
    labels = torch.tensor([0, 1, 2, 1], device=device)
    criterion = nn.CrossEntropyLoss().to(device)

    model.train()
    with autocast(device_type=device.type, enabled=(device.type == "cuda")):
        out = model(dummy)
        loss = criterion(out, labels)
    loss.backward()
    print(f"[✓] Forward pass  — output shape: {out.shape}")
    print(f"[✓] Backward pass — loss: {loss.item():.4f}")

    # Dummy ROC-AUC
    dummy_logits = torch.randn(100, 3)
    dummy_labels = torch.randint(0, 3, (100,))

    tmpdir = "./output"
    os.makedirs(tmpdir, exist_ok=True)

    class _FakeCfg:
        output_dir = tmpdir

    auc_score = evaluate_roc_auc(dummy_logits, dummy_labels, _FakeCfg())
    print(f"[✓] ROC-AUC computed: {auc_score:.4f}")

    print("\n" + "=" * 60)
    print("  DRY RUN PASSED — pipeline is functional ✓")
    print("=" * 60)


# ──────────────────────────────────────────────────────────────────────────────
# 9. Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    cfg = get_config()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # ---- dry run check ----
    if getattr(cfg, "dry_run", False):
        dry_run()
        return

    # ---- extract data on Kaggle ----
    extract_dataset(cfg)

    # ---- dataset & loaders ----
    full_dataset = LensingDataset(
        root=cfg.data_dir,
        transform=get_transforms(cfg, train=True),
    )

    val_size = int(len(full_dataset) * cfg.val_split)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg.seed),
    )

    # Override transform for val split (no augmentation)
    val_ds.dataset = LensingDataset(
        root=cfg.data_dir,
        transform=get_transforms(cfg, train=False),
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )

    print(f"\n[INFO] Train samples: {train_size}  |  Val samples: {val_size}")

    # ---- model ----
    model, device = build_model(num_classes=3)

    # ---- train ----
    history, val_logits, val_labels = train(
        model, train_loader, val_loader, cfg, device
    )

    # ---- evaluate ----
    macro_auc = evaluate_roc_auc(val_logits, val_labels, cfg)

    print(f"\n{'='*50}")
    print(f"  Training complete — Final Macro ROC-AUC: {macro_auc:.4f}")
    print(f"  Model saved to:  {os.path.join(cfg.output_dir, 'model.pth')}")
    print(f"  ROC curve at:    {os.path.join(cfg.output_dir, 'roc_curve.png')}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()

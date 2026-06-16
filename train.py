"""
train.py

Trains a building segmentation model from a YAML config.
Saves the best checkpoint (by val IoU) and a last checkpoint after every epoch.
Produces a loss/IoU plot at the end.

Usage:
    python train.py --config configs/phase2_segformer_mitb5.yaml
    python train.py --config configs/phase2_segformer_mitb5.yaml \
                    --resume checkpoints/phase2_segformer_mitb5_last.pth
"""

import os
import yaml
import torch
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from src.model import BuildingSegmentationModel
from src.dataset import PatchDataset
from src.losses import combined_loss
from src.utils import set_seed, calculate_iou, split_patches_by_image


def main():
    parser = argparse.ArgumentParser(description="Building segmentation training pipeline")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--resume", default=None,  help="Checkpoint path to resume from (.pth)")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed()

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    os.makedirs("checkpoints", exist_ok=True)

    print(f"experiment : {cfg['experiment_name']}")
    print(f"device     : {device}")

    full_train_ds = PatchDataset(
        image_dir=cfg["dataset"]["image_dir"],
        gt_dir=cfg["dataset"]["gt_dir"],
        patch_size=cfg["dataset"]["patch_size"],
        stride=cfg["dataset"]["stride"],
        mode="train",
    )
    full_val_ds = PatchDataset(
        image_dir=cfg["dataset"]["image_dir"],
        gt_dir=cfg["dataset"]["gt_dir"],
        patch_size=cfg["dataset"]["patch_size"],
        stride=cfg["dataset"]["stride"],
        mode="val",
    )

    train_idx, val_idx = split_patches_by_image(
        full_train_ds.patches, test_size=cfg["dataset"]["val_size"]
    )
    train_dataset = Subset(full_train_ds, train_idx)
    val_dataset = Subset(full_val_ds,   val_idx)

    print(f"train patches : {len(train_dataset)}")
    print(f"val patches   : {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True, num_workers=8, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False, num_workers=8, pin_memory=True,
    )

    model = BuildingSegmentationModel(
        architecture=cfg["model"]["architecture"],
        encoder_name=cfg["model"]["encoder_name"],
        encoder_weights=cfg["model"]["encoder_weights"],
    ).to(device)

    optimizer = AdamW(model.parameters(),
                      lr=cfg["training"]["lr"],
                      weight_decay=cfg["training"]["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["training"]["epochs"])
    scaler = GradScaler(enabled=cfg["training"]["mixed_precision"])

    start_epoch = 1
    best_val_iou = 0.0
    train_losses = []
    val_ious = []

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"resuming from {args.resume}")
            ckpt = torch.load(args.resume, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            best_val_iou = ckpt.get("best_val_iou", 0.0)
            train_losses = ckpt.get("train_losses", [])
            val_ious = ckpt.get("val_ious", [])
            if "scheduler_state_dict" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            else:
                for _ in range(1, start_epoch):
                    scheduler.step()
            if "scaler_state_dict" in ckpt:
                scaler.load_state_dict(ckpt["scaler_state_dict"])
        else:
            print(f"checkpoint not found: {args.resume}, starting from scratch")

    for epoch in range(start_epoch, cfg["training"]["epochs"] + 1):
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{cfg['training']['epochs']} [train]")
        for imgs, masks in pbar:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            with autocast(enabled=cfg["training"]["mixed_precision"]):
                loss = combined_loss(model(imgs), masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()

        model.eval()
        val_iou = 0.0
        with torch.no_grad():
            for imgs, masks in tqdm(val_loader, desc=f"epoch {epoch}/{cfg['training']['epochs']} [val]"):
                imgs, masks = imgs.to(device), masks.to(device)
                val_iou += calculate_iou(model(imgs), masks)

        avg_iou = val_iou / len(val_loader)
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        val_ious.append(avg_iou)

        print(f"epoch {epoch} | loss {avg_loss:.4f} | val iou {avg_iou:.4f}")

        ckpt_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_val_iou": best_val_iou,
            "train_losses": train_losses,
            "val_ious": val_ious,
        }
        torch.save(ckpt_data, f"checkpoints/{cfg['experiment_name']}_last.pth")

        if avg_iou > best_val_iou:
            best_val_iou = avg_iou
            best_path = f"checkpoints/{cfg['experiment_name']}_best.pth"
            torch.save(ckpt_data, best_path)
            print(f"  saved best checkpoint (iou {best_val_iou:.4f}) -> {best_path}")

    # Plot training curves
    epochs_range = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs_range, train_losses, color="red",  marker="o", label="Train loss")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.grid(True)
    ax1.legend()

    ax2.plot(epochs_range, val_ious, color="blue", marker="s", label="Val IoU")
    ax2.set_title("IoU")
    ax2.set_xlabel("Epoch")
    ax2.grid(True)
    ax2.legend()

    plot_path = f"checkpoints/{cfg['experiment_name']}_metrics_plot.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    print(f"training plot saved -> {plot_path}")


if __name__ == "__main__":
    main()

"""
scripts/evaluate.py

Evaluates a trained model on the validation split.
Reproduces the same spatial split used during training.

Usage:
    python -m scripts.evaluate \
        --config configs/phase2_segformer_mitb5.yaml \
        --checkpoint checkpoints/phase2_segformer_mitb5_best.pth [--tta]
"""
import os
import argparse
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset

from src.dataset import PatchDataset
from src.utils import split_patches_by_image
from src.inference import build_model, predict_patch_probs

os.environ["OPENCV_LOG_LEVEL"] = "FATAL"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--empty_eq_one", action="store_true")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device} | tta: {args.tta}")

    full_val_ds = PatchDataset(
        image_dir=cfg["dataset"]["image_dir"],
        gt_dir=cfg["dataset"]["gt_dir"],
        patch_size=cfg["dataset"]["patch_size"],
        stride=cfg["dataset"]["stride"],
        mode="val",
    )
    _, val_idx = split_patches_by_image(
        full_val_ds.patches,
        test_size=cfg["dataset"]["val_size"],
        random_state=42,
    )
    val_dataset = Subset(full_val_ds, val_idx)
    print(f"val patches: {len(val_dataset)}")

    batch_size = args.batch_size or cfg["training"]["batch_size"]
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=4, pin_memory=True,
    )

    model, ckpt = build_model(cfg, args.checkpoint, device)
    epoch = ckpt.get("epoch", "?")
    print(f"checkpoint: epoch {epoch}")

    thresholds = np.round(np.arange(0.30, 0.71, 0.05), 2)
    inter_acc = {float(t): 0.0 for t in thresholds}
    union_acc = {float(t): 0.0 for t in thresholds}
    tp_acc = {float(t): 0.0 for t in thresholds}
    fp_acc = {float(t): 0.0 for t in thresholds}
    fn_acc = {float(t): 0.0 for t in thresholds}

    ft = args.threshold
    pp_sum, pp_n = 0.0, 0
    pp_ne_sum, pp_ne_n = 0.0, 0

    print("evaluating...")
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            target = (masks > 0.5).float()
            probs = predict_patch_probs(model, imgs, tta=args.tta)

            for t in thresholds:
                pred = (probs > t).float()
                inter = (pred * target).sum().item()
                union = (pred + target - pred * target).sum().item()
                inter_acc[float(t)] += inter
                union_acc[float(t)] += union
                tp_acc[float(t)] += inter
                fp_acc[float(t)] += (pred * (1 - target)).sum().item()
                fn_acc[float(t)] += ((1 - pred) * target).sum().item()

            pred_ft = (probs > ft).float()
            dims = (1, 2, 3)
            inter_pp = (pred_ft * target).sum(dim=dims)
            union_pp = (pred_ft + target - pred_ft * target).sum(dim=dims)
            target_sum = target.sum(dim=dims)

            for j in range(pred_ft.shape[0]):
                u = union_pp[j].item()
                iou_j = (1.0 if args.empty_eq_one else 0.0) if u == 0 \
                    else inter_pp[j].item() / u
                pp_sum += iou_j
                pp_n += 1
                if target_sum[j].item() > 0:
                    pp_ne_sum += iou_j
                    pp_ne_n += 1

    print("\nthreshold sweep (dataset IoU):")
    best_t, best_iou = ft, -1.0
    for t in thresholds:
        t = float(t)
        iou = inter_acc[t] / (union_acc[t] + 1e-6)
        if iou > best_iou:
            best_iou, best_t = iou, t
        print(f"  {t:.2f}  {iou:.4f}")
    print(f"  best: {best_t:.2f}  {best_iou:.4f}")

    ds_iou = inter_acc[ft] / (union_acc[ft] + 1e-6)
    tp, fp, fn = tp_acc[ft], fp_acc[ft], fn_acc[ft]
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    mean_pp = pp_sum / max(pp_n, 1)
    mean_pp_ne = pp_ne_sum / max(pp_ne_n, 1)

    print(f"\nmetrics @ threshold {ft}:")
    print(f"dataset IoU {ds_iou:.4f}")
    print(f"mean-patch IoU {mean_pp:.4f}")
    print(f"mean-patch IoU* {mean_pp_ne:.4f}  (* non-empty patches: {pp_ne_n}/{pp_n})")
    print(f"precision {precision:.4f}")
    print(f"recall {recall:.4f}")
    print(f"F1 {f1:.4f}")


if __name__ == "__main__":
    main()

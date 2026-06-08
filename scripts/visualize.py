"""
scripts/visualize.py

Visualizes model predictions in three modes:
  predict - 3-panel grid: image / confidence map / binary mask
  pipeline - 5-panel postprocessing steps per patch
  fulltile - full tile inference with polygon overlay

Usage:
    python -m scripts.visualize --mode predict \
        --config configs/phase2_segformer_mitb5.yaml \
        --checkpoint checkpoints/phase2_segformer_mitb5_best.pth

    python -m scripts.visualize --mode fulltile \
        --config configs/phase2_segformer_mitb5.yaml \
        --checkpoint checkpoints/phase2_segformer_mitb5_best.pth \
        --tile data/inria/train/images/austin1.tif \
        --gt data/inria/train/gt/austin1.tif
"""
import os
import random
import argparse

import cv2
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Subset

from src.dataset import PatchDataset
from src.utils import split_patches_by_image
from src.postprocess import postprocess_pipeline, extract_building_polygons
from src.inference import build_model, denormalize, predict_patch_probs, infer_full_tile


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["predict", "pipeline", "fulltile"], required=True)
    p.add_argument("--config",     required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out_dir",    default="predictions")
    p.add_argument("--threshold",  type=float, default=0.5)
    p.add_argument("--tta",        action="store_true")
    p.add_argument("--num_samples", type=int, default=5)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--tile",        type=str)
    p.add_argument("--gt",          type=str)
    p.add_argument("--overlap",     type=int, default=128)
    p.add_argument("--batch_size",  type=int, default=8)
    p.add_argument("--display_max", type=int, default=1600)
    return p.parse_args()


def load_model(args):
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = build_model(cfg, args.checkpoint, device)
    return cfg, device, model


def get_val_subset(cfg):
    ds = PatchDataset(
        image_dir=cfg["dataset"]["image_dir"],
        gt_dir=cfg["dataset"]["gt_dir"],
        patch_size=cfg["dataset"]["patch_size"],
        stride=cfg["dataset"]["stride"],
        mode="val",
    )
    _, idx = split_patches_by_image(
        ds.patches, test_size=cfg["dataset"]["val_size"], random_state=42
    )
    return Subset(ds, idx)


def sample_indices(n, k, seed):
    random.seed(seed)
    return random.sample(range(n), min(k, n))


def run_predict(args, cfg, device, model):
    val = get_val_subset(cfg)
    print(f"val patches: {len(val)}")

    for i, idx in enumerate(sample_indices(len(val), args.num_samples, args.seed), 1):
        img_t, _ = val[idx]
        prob = predict_patch_probs(model, img_t.unsqueeze(0).to(device), args.tta)
        prob = prob.squeeze().cpu().numpy()
        mask = (prob > args.threshold).astype(np.uint8)
        orig = denormalize(img_t)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(orig);  axes[0].set_title("Image");      axes[0].axis("off")
        im = axes[1].imshow(prob, cmap="jet", vmin=0, vmax=1)
        axes[1].set_title("Confidence"); axes[1].axis("off")
        fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        axes[2].imshow(mask, cmap="gray")
        axes[2].set_title(f"Mask (t={args.threshold})"); axes[2].axis("off")
        plt.tight_layout()

        path = os.path.join(args.out_dir, f"predict_{i}.png")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  saved {path}")


def run_pipeline(args, cfg, device, model):
    val = get_val_subset(cfg)

    for i, idx in enumerate(sample_indices(len(val), args.num_samples, seed=123), 1):
        img_t, _ = val[idx]
        prob = predict_patch_probs(model, img_t.unsqueeze(0).to(device), args.tta)
        raw  = (prob.squeeze().cpu().numpy() > args.threshold).astype(np.uint8)
        morph, approx = postprocess_pipeline(raw)
        orig = denormalize(img_t)

        overlay = orig.copy()
        green = np.zeros_like(orig)
        green[approx == 255] = [0, 1, 0]
        cv2.addWeighted(green, 0.4, overlay, 1.0, 0, overlay)

        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        axes[0].imshow(orig);    axes[0].set_title("Image",        fontweight="bold")
        axes[1].imshow(raw,   cmap="gray"); axes[1].set_title("Raw mask")
        axes[2].imshow(morph, cmap="gray"); axes[2].set_title("Morphology")
        axes[3].imshow(approx,cmap="gray"); axes[3].set_title("Douglas-Peucker")
        axes[4].imshow(overlay);            axes[4].set_title("Overlay", fontweight="bold")
        for a in axes:
            a.axis("off")
        plt.tight_layout()

        path = os.path.join(args.out_dir, f"pipeline_{i}.png")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  saved {path}")


def _downscale(img, max_side):
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s >= 1.0:
        return img
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


def run_fulltile(args, cfg, device, model):
    if not args.tile or not os.path.isfile(args.tile):
        raise FileNotFoundError(f"tile not found: {args.tile}")

    patch = cfg["dataset"]["patch_size"]
    img_bgr = cv2.imread(args.tile, cv2.IMREAD_COLOR)
    print(f"tile: {os.path.basename(args.tile)} ({img_bgr.shape[1]}x{img_bgr.shape[0]})")

    prob = infer_full_tile(model, img_bgr, patch, args.overlap, device,
                             batch_size=args.batch_size, tta=args.tta)
    binary = (prob > args.threshold).astype(np.uint8) * 255
    polys = extract_building_polygons(binary, min_area=40, simplify_frac=0.01)

    orig_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    overlay = orig_rgb.copy()
    green = np.zeros_like(orig_rgb)
    for poly, _ in polys:
        cv2.fillPoly(green, [poly.astype(np.int32)], (0, 255, 0))
    cv2.addWeighted(green, 0.6, overlay, 1.0, 0, overlay)

    stem = os.path.splitext(os.path.basename(args.tile))[0]
    cv2.imwrite(os.path.join(args.out_dir, f"{stem}_mask.png"), binary)
    cv2.imwrite(os.path.join(args.out_dir, f"{stem}_overlay.png"),
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    panels = [(_downscale(orig_rgb, args.display_max), "Image")]
    if args.gt and os.path.isfile(args.gt):
        gt = cv2.imread(args.gt, cv2.IMREAD_GRAYSCALE)
        panels.append((_downscale(gt, args.display_max), "Ground truth"))
    panels.append((_downscale(binary,  args.display_max), "Predicted mask"))
    panels.append((_downscale(overlay, args.display_max), f"Overlay ({len(polys)} buildings)"))

    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 6))
    for ax, (im, title) in zip(axes, panels):
        ax.imshow(im, cmap="gray" if im.ndim == 2 else None)
        ax.set_title(title, fontsize=12)
        ax.axis("off")
    plt.tight_layout()

    cmp = os.path.join(args.out_dir, f"{stem}_compare.png")
    plt.savefig(cmp, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"buildings: {len(polys)}")
    print(f"saved: {stem}_mask.png, {stem}_overlay.png, {stem}_compare.png -> {args.out_dir}/")


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    cfg, device, model = load_model(args)
    print(f"mode: {args.mode} | device: {device} | tta: {args.tta}")

    if args.mode == "predict":
        run_predict(args, cfg, device, model)
    elif args.mode == "pipeline":
        run_pipeline(args, cfg, device, model)
    else:
        run_fulltile(args, cfg, device, model)


if __name__ == "__main__":
    main()

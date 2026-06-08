"""
tools/extract_pth.py

Prints training metrics stored inside a checkpoint file.

Usage:
    python -m tools.extract_pth
"""
import torch

CHECKPOINT = "checkpoints/phase2_segformer_mitb5_best.pth"

ckpt = torch.load(CHECKPOINT, map_location="cpu")

print(f"checkpoint: {CHECKPOINT}")
print(f"epoch: {ckpt['epoch']}")
print(f"best IoU: {ckpt['best_val_iou']:.4f}")

print("\nval IoU per epoch:")
for i, iou in enumerate(ckpt["val_ious"], 1):
    print(f"{i:>3}  {iou:.4f}")

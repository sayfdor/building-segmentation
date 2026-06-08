"""
src/inference.py

Shared inference logic used by scripts/visualize.py and scripts/evaluate.py.
Provides model loading, denormalization, patch-level inference with optional TTA,
and  full-tile sliding-window inference with Hann-window blending.

Note: the model was trained on images read by cv2 (BGR order) without
RGB conversion. Inference must follow the same convention. The RGB swap in
denormalize() is applied only for visualization purposes.
"""

import numpy as np
import torch

from src.model import BuildingSegmentationModel

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def build_model(cfg, checkpoint_path, device):
    model = BuildingSegmentationModel(
        architecture=cfg["model"]["architecture"],
        encoder_name=cfg["model"]["encoder_name"],
        encoder_weights=None,
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def denormalize(tensor_img):
    img = tensor_img.permute(1, 2, 0).cpu().numpy()
    img = img * STD + MEAN
    img = np.clip(img, 0, 1)
    return np.ascontiguousarray(img[:, :, ::-1])


@torch.no_grad()
def predict_patch_probs(model, x, tta=False):
    if not tta:
        return torch.sigmoid(model(x))

    outs = []
    for k in range(4):
        xr = torch.rot90(x, k, dims=(2, 3))
        outs.append(torch.rot90(torch.sigmoid(model(xr)), -k, dims=(2, 3)))
    outs.append(torch.flip(torch.sigmoid(model(torch.flip(x, dims=(3,)))), dims=(3,)))
    outs.append(torch.flip(torch.sigmoid(model(torch.flip(x, dims=(2,)))), dims=(2,)))
    return torch.stack(outs).mean(0)


def build_window(size):
    h = np.hanning(size).astype(np.float32)
    return np.outer(h, h) * 0.9 + 0.1


def window_positions(length, patch, stride):
    if length <= patch:
        return [0]
    pos = list(range(0, length - patch + 1, stride))
    if pos[-1] != length - patch:
        pos.append(length - patch)
    return pos


@torch.no_grad()
def infer_full_tile(model, img_bgr, patch, overlap, device, batch_size=8, tta=False):
    stride = patch - overlap
    assert stride > 0, "overlap must be less than patch size"
    window = build_window(patch)
    H, W = img_bgr.shape[:2]

    img = img_bgr.astype(np.float32) / 255.0
    img = (img - MEAN) / STD

    prob_accum   = np.zeros((H, W), dtype=np.float32)
    weight_accum = np.zeros((H, W), dtype=np.float32)

    ys = window_positions(H, patch, stride)
    xs = window_positions(W, patch, stride)
    coords, tensors = [], []

    def flush():
        if not tensors:
            return
        batch = torch.from_numpy(np.stack(tensors)).to(device)
        probs = predict_patch_probs(model, batch, tta)[:, 0].cpu().numpy()
        for (yy, xx), pr in zip(coords, probs):
            prob_accum[yy:yy + patch, xx:xx + patch]   += pr * window
            weight_accum[yy:yy + patch, xx:xx + patch] += window
        coords.clear()
        tensors.clear()

    for y in ys:
        for x in xs:
            tensors.append(np.transpose(img[y:y + patch, x:x + patch, :], (2, 0, 1)))
            coords.append((y, x))
            if len(tensors) >= batch_size:
                flush()
    flush()

    return prob_accum / np.maximum(weight_accum, 1e-6)

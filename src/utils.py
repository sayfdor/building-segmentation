import os
import random
from pathlib import Path
from collections import defaultdict

import torch
from sklearn.model_selection import train_test_split


def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_iou(pred, target, threshold: float = 0.5):
    with torch.no_grad():
        pred_prob = torch.sigmoid(pred)
        pred_bin = (pred_prob > threshold).float()
        target_bin = (target > 0.5).float()
        intersection = (pred_bin * target_bin).sum(dim=(1, 2, 3))
        union = (pred_bin + target_bin - pred_bin * target_bin).sum(dim=(1, 2, 3))
        return (intersection / (union + 1e-6)).mean().item()


def split_patches_by_image(patches, test_size: float = 0.2, random_state: int = 42):
    img_to_idx = defaultdict(list)
    for i, (img_path, _, _, _) in enumerate(patches):
        img_to_idx[Path(img_path).name].append(i)

    img_names = list(img_to_idx.keys())
    train_imgs, val_imgs = train_test_split(
        img_names, test_size=test_size, random_state=random_state
    )

    train_indices = []
    for img in train_imgs:
        train_indices.extend(img_to_idx[img])

    val_indices = []
    for img in val_imgs:
        val_indices.extend(img_to_idx[img])

    return train_indices, val_indices

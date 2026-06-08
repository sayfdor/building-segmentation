import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

os.environ["OPENCV_LOG_LEVEL"] = "FATAL"


class PatchDataset(Dataset):
    def __init__(self, image_dir: str, gt_dir: str, patch_size: int = 512, stride: int = 430, mode: str = 'train'):
        self.patch_size = patch_size
        self.stride = stride
        self.mode = mode
        self.patches = []

        image_paths = sorted(Path(image_dir).glob("*.tif"))
        gt_paths = sorted(Path(gt_dir).glob("*.tif"))
        gt_map = {p.name: p for p in gt_paths}

        self._init_augmentations()

        for img_path in image_paths:
            if img_path.name not in gt_map:
                continue
            gt_path = gt_map[img_path.name]

            try:
                with Image.open(img_path) as img:
                    w, h = img.size
            except Exception:
                continue

            for y in range(0, h - patch_size + 1, stride):
                for x in range(0, w - patch_size + 1, stride):
                    self.patches.append((str(img_path), str(gt_path), x, y))

            if (h - patch_size) % stride != 0:
                y = h - patch_size
                for x in range(0, w - patch_size + 1, stride):
                    self.patches.append((str(img_path), str(gt_path), x, y))
            if (w - patch_size) % stride != 0:
                x = w - patch_size
                for y in range(0, h - patch_size + 1, stride):
                    self.patches.append((str(img_path), str(gt_path), x, y))

    def _init_augmentations(self):
        self.train_aug = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        self.val_aug = A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        img_path, mask_path, x, y = self.patches[idx]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            img = np.zeros((self.patch_size, self.patch_size, 3), dtype=np.uint8)
            mask = np.zeros((self.patch_size, self.patch_size), dtype=np.uint8)
        else:
            img = img[y:y + self.patch_size, x:x + self.patch_size]
            mask = mask[y:y + self.patch_size, x:x + self.patch_size]

        mask = (mask > 127).astype(np.uint8)
        aug = self.train_aug if self.mode == 'train' else self.val_aug
        augmented = aug(image=img, mask=mask)

        return augmented['image'], augmented['mask'].long().unsqueeze(0)
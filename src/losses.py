import torch
import torch.nn.functional as F


def dice_loss(logits, target, smooth=1e-6):
    pred = torch.sigmoid(logits)
    pred = pred.float().contiguous().view(-1)
    target = target.float().contiguous().view(-1)
    intersection = (pred * target).sum()
    return 1 - (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def focal_loss(logits, target, alpha=0.25, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(logits, target.float(), reduction="none")
    pt = torch.exp(-bce)
    loss = alpha * (1 - pt) ** gamma * bce
    return loss.mean()


def morphological_gradient(mask, kernel_size=3):
    device = mask.device
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=device)
    dilated = F.conv2d(mask.float(), kernel, padding=kernel_size // 2)
    dilated = (dilated > 0).float()
    eroded = F.conv2d(mask.float(), kernel, padding=kernel_size // 2)
    eroded = (eroded == kernel.sum()).float()
    return (dilated - eroded).clamp(min=0)


def combined_loss(logits, target, alpha=0.4, beta=0.4, gamma=0.2,
                  kernel_size=3, smoothing=0.05):
    logits = logits.float()
    target = target.float()
    target_smooth = target * (1.0 - smoothing) + smoothing * 0.5

    dice = dice_loss(logits, target_smooth)
    focal = focal_loss(logits, target_smooth)

    boundary = morphological_gradient(target, kernel_size=kernel_size)
    bce_map = F.binary_cross_entropy_with_logits(logits, target_smooth, reduction="none")
    boundary_loss = (bce_map * boundary).sum() / (boundary.sum() + 1e-6)

    return alpha * dice + beta * focal + gamma * boundary_loss

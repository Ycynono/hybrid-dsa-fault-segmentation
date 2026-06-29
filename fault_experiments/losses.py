import torch
import torch.nn as nn


class DiceBCELoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1e-6):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        dims = tuple(range(1, probs.ndim))
        intersection = torch.sum(probs * targets, dim=dims)
        denominator = torch.sum(probs + targets, dim=dims)
        dice_loss = 1.0 - torch.mean((2.0 * intersection + self.smooth) / (denominator + self.smooth))
        return self.bce_weight * bce + self.dice_weight * dice_loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        loss = alpha_t * (1.0 - pt).pow(self.gamma) * bce
        return loss.mean()


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        dims = tuple(range(1, probs.ndim))
        tp = torch.sum(probs * targets, dim=dims)
        fp = torch.sum(probs * (1.0 - targets), dim=dims)
        fn = torch.sum((1.0 - probs) * targets, dim=dims)
        score = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - score.mean()


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, gamma=0.75, smooth=1e-6):
        super().__init__()
        self.tversky = TverskyLoss(alpha=alpha, beta=beta, smooth=smooth)
        self.gamma = gamma

    def forward(self, logits, targets):
        return self.tversky(logits, targets).pow(self.gamma)


class HybridFocalTverskyLoss(nn.Module):
    def __init__(self, focal_weight=0.4, tversky_weight=0.6, focal_alpha=0.75, focal_gamma=2.0):
        super().__init__()
        self.focal_weight = focal_weight
        self.tversky_weight = tversky_weight
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.tversky = TverskyLoss(alpha=0.3, beta=0.7)

    def forward(self, logits, targets):
        return self.focal_weight * self.focal(logits, targets) + self.tversky_weight * self.tversky(
            logits, targets
        )


def build_loss(name):
    name = name.lower()
    if name in {"dice_bce", "dice+bce", "bce_dice"}:
        return DiceBCELoss(bce_weight=0.4, dice_weight=0.6)
    if name in {"focal", "focal_loss"}:
        return FocalLoss()
    if name in {"tversky", "tversky_loss"}:
        return TverskyLoss()
    if name in {"focal_tversky", "focal-tversky"}:
        return FocalTverskyLoss()
    if name in {"hybrid_focal_tversky", "hybrid"}:
        return HybridFocalTverskyLoss()
    raise ValueError(f"Unknown loss name: {name}")

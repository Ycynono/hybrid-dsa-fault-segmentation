import torch


def binary_stats_from_logits(logits, targets, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = probs >= threshold
    targets = targets >= 0.5

    tp = torch.logical_and(preds, targets).sum().float()
    fp = torch.logical_and(preds, ~targets).sum().float()
    fn = torch.logical_and(~preds, targets).sum().float()
    tn = torch.logical_and(~preds, ~targets).sum().float()
    return tp, fp, fn, tn


def binary_segmentation_metrics(logits, targets, threshold=0.5, eps=1e-7):
    tp, fp, fn, tn = binary_stats_from_logits(logits, targets, threshold)
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    specificity = (tn + eps) / (tn + fp + eps)
    accuracy = (tp + tn + eps) / (tp + tn + fp + fn + eps)
    return {
        "dice": float(dice.detach().cpu()),
        "iou": float(iou.detach().cpu()),
        "precision": float(precision.detach().cpu()),
        "recall": float(recall.detach().cpu()),
        "specificity": float(specificity.detach().cpu()),
        "accuracy": float(accuracy.detach().cpu()),
    }


class MetricAverager:
    def __init__(self):
        self.totals = {}
        self.count = 0

    def update(self, metrics):
        self.count += 1
        for key, value in metrics.items():
            self.totals[key] = self.totals.get(key, 0.0) + float(value)

    def compute(self):
        if self.count == 0:
            return {}
        return {key: value / self.count for key, value in self.totals.items()}

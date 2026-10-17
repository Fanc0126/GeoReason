"""Point-sampled segmentation objectives."""
import torch
from torch.nn import functional as F


def point_sample(values, coordinates):
    return F.grid_sample(values[:, None].float(), coordinates[:, :, None] * 2 - 1,
                         align_corners=False)[:, 0, :, 0]


def mask_losses(logits, targets, num_points=12544, oversample=3, importance=0.75):
    n = logits.shape[0]
    with torch.no_grad():
        coordinates = torch.rand(n, num_points * oversample, 2, device=logits.device)
        uncertainty = -point_sample(logits, coordinates).abs()
        count = int(num_points * importance)
        indices = uncertainty.topk(count, dim=1).indices
        important = coordinates.gather(1, indices[:, :, None].expand(-1, -1, 2))
        coordinates = torch.cat([important, torch.rand(n, num_points - count, 2, device=logits.device)], 1)
        targets = point_sample(targets, coordinates)
    predictions = point_sample(logits, coordinates)
    bce = F.binary_cross_entropy_with_logits(predictions, targets, reduction="sum") / (n * num_points + 1e-4)
    probabilities = predictions.sigmoid()
    dice = 1 - (2 * (probabilities * targets).sum(1) + 1) / (probabilities.sum(1) + targets.sum(1) + 1)
    return 2 * bce, 0.5 * dice.sum() / (n + 1e-4)

import torch

eps = 1e-6

def soft_dice_loss(outputs, targets, per_image=False, per_channel=False):
    batch_size, n_channels = outputs.size(0), outputs.size(1)
    
    eps = 1e-6
    n_parts = 1
    if per_image:
        n_parts = batch_size
    if per_channel:
        n_parts = batch_size * n_channels
    
    dice_target = targets.contiguous().view(n_parts, -1).float()
    dice_output = outputs.contiguous().view(n_parts, -1)
    intersection = torch.sum(dice_output * dice_target, dim=1)
    union = torch.sum(dice_output, dim=1) + torch.sum(dice_target, dim=1) + eps
    loss = (1 - (2 * intersection + eps) / union).mean()
    return loss

def dice_metric(preds, trues, per_image=False, per_channel=False):
    preds = preds.float()
    return 1 - soft_dice_loss(preds, trues, per_image, per_channel)
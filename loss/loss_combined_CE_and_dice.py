import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)
    # 只考虑前景
    input_f = input[:, 1]
    target_f = target[:, 1]
    inter = 2 * (input_f * target_f).sum(dim=sum_dim)
    sets_sum = input_f.sum(dim=sum_dim) + target_f.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def dice_coeff_single_mask(input: Tensor, target: Tensor, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 2

    inter = 2 * (input * target).sum()
    sets_sum = input.sum() + target.sum()
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    # return 1 - fn(input, target, reduce_batch_first=True)
    return 1 - fn(input, target, reduce_batch_first=False)


class LossCombinedCEAndDice(nn.Module):
    def __init__(self, classes, rate):
        super().__init__()
        self.classes = classes
        if classes == 1:
            self.ce_loss = nn.BCEWithLogitsLoss()
        else:
            self.ce_loss = nn.CrossEntropyLoss()
        self.rate = rate

    def forward(self, pred, target):
        loss = 0
        if self.classes == 1:
            loss = self.ce_loss(pred, target)
            loss = (1 - self.rate) * loss + self.rate * dice_loss(F.sigmoid(pred.squeeze()), target.squeeze(),
                                                                  multiclass=False)
        else:
            loss = self.ce_loss(pred, target.squeeze(1))
            loss = (1 - self.rate) * loss + self.rate * dice_loss(
                F.softmax(pred, dim=1).float(),
                F.one_hot(target.squeeze(1), self.classes).permute(0, 3, 1, 2).float(),
                multiclass=False
            )
        return loss

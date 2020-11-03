import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import reduce_loss


@LOSSES.register_module()
class DiceLoss(nn.Module):

    def __init__(self,
                 optimize_bg=True,
                 smooth=1.0,
                 mode='linear',
                 reduction='mean',
                 loss_weight=1.0,
                 class_weight=None):

        super(DiceLoss, self).__init__()
        self.optimize_bg = optimize_bg
        self.smooth = smooth
        self.mode = mode
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=255,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        cls_score = F.softmax(cls_score, dim=1)
        b, num_class = cls_score.shape[:2]
        cls_score = cls_score.view(b, num_class, -1)
        # label = label.view(b, 1, -1)
        cls_score = cls_score.permute(1, 0, 2).contiguous().view(num_class, -1)
        label = label.view(1, -1)
        mask = (label != ignore_index)
        label = label[mask]
        label = label.view(1, -1)
        cls_score = cls_score[mask.repeat(num_class, 1)].view(num_class, -1)

        one_hot_label = torch.zeros_like(cls_score)
        one_hot_label.scatter_(0, label, 1)

        if weight is not None:
            weight = weight.float()
            weight = weight.view(1, -1)
            weight = weight[mask]
            weight = weight.view(1, -1)
            one_hot_label = one_hot_label * weight
        # pdb.set_trace()
        intersection = (cls_score * one_hot_label).sum(-1) + self.smooth
        # print(intersection.shape)
        # area_cls = cls_score.sum(0)
        # area_label = one_hot_label.sum(0)
        area_cls = (cls_score * cls_score).sum(-1) + self.smooth
        area_label = (one_hot_label * one_hot_label).sum(-1) + self.smooth
        dice = (2 * intersection) / (area_cls + area_label)
        if self.mode == 'log':
            dice_loss = torch.neg(torch.log(dice))
        elif self.mode == 'linear':
            dice_loss = 1 - dice
        elif self.mode == 'focal':
            dice_loss = (1 - dice) * (1 - dice)
        elif self.mode == 'focal1':
            dice_loss = (1 - dice)**(1 - dice) * (1 - dice)
        elif self.mode == 'focal2':
            dice_loss = (1 - dice)**dice * (1 - dice)
        elif self.mode == 'focal3':
            dice_loss = (1 - dice)**(1 / dice) * (1 - dice)

        if self.class_weight is not None:
            class_weight = self.class_weight.float()
            class_weight = class_weight.view(1, -1)
            dice_loss = dice_loss * class_weight

        if not self.optimize_bg:
            dice_loss = dice_loss[1:]

        dice_loss = self.loss_weight * reduce_loss(dice_loss, reduction)
        return dice_loss

import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Union


class Loss(nn.Module):
    def __init__(self, loss_function="WCE"):
        super(Loss, self).__init__()
        self.name = loss_function

    def forward(self, preds, labels):
        if self.name == "BCE":
            final_loss = bce_loss(preds, labels)
        elif self.name == "WCE":
            final_loss = weighted_bce_loss(preds, labels)
        elif self.name == "Dice":
            final_loss = 0.001 * bce_loss(preds, labels) + dice_loss(preds, labels)
        elif self.name == "SSIM":
            final_loss = ssim_loss(preds, labels) + 0.001 * bce_loss(preds, labels)
        elif self.name == "IOU":
            final_loss = iou_loss(preds, labels) + 0.1 * weighted_bce_loss(preds, labels)
        elif self.name == "Tversky":
            final_loss = tversky_loss(preds, labels) + 0.1 * weighted_bce_loss(preds, labels)
        elif self.name == "FTL++":
            final_loss = focal_tversky_loss_plus_plus(preds, labels)
        elif self.name == "HFL":
            final_loss = focal_tversky_loss_plus_plus(preds, labels) + 0.001 * focal_loss(preds, labels)
        elif self.name == "NFL":
            final_loss = (
                focal_tversky_loss_plus_plus(preds, labels)
                + 1 * normalize_focalloss(preds, labels)
                + grad_loss(preds, labels)
            )
        elif self.name == "ARST":
            final_loss = focal_tversky_loss_plus_plus(
                preds, labels
            ) + 0.1 * Binary_Adaptive_Region_Specific_TverskyLoss()(preds, labels)
        else:
            raise NameError

        return final_loss


def bce_loss(preds, labels):
    bce_loss = F.binary_cross_entropy(preds, labels, reduction="sum")

    return bce_loss


def weighted_bce_loss(preds, labels):
    beta = 1 - torch.mean(labels)
    weights = 1 - beta + (2 * beta - 1) * labels
    wce_loss = F.binary_cross_entropy(preds, labels, weights, reduction="sum")

    return wce_loss


def dice_loss(preds, labels):
    n = preds.size(0)
    dice_loss = 0.0
    for i in range(n):
        prob_2 = torch.mul(preds[i, :, :, :], preds[i, :, :, :])
        label_2 = torch.mul(labels[i, :, :, :], labels[i, :, :, :])
        prob_label = torch.mul(preds[i, :, :, :], labels[i, :, :, :])
        sum_prob_2 = torch.sum(prob_2)
        sum_label_2 = torch.sum(label_2)
        sum_prob_label = torch.sum(prob_label)
        sum_prob_label = sum_prob_label + 0.000001
        temp_loss = (sum_prob_2 + sum_label_2) / (2 * sum_prob_label)
        if temp_loss.data.item() > 50:
            temp_loss = 50
        dice_loss = dice_loss + temp_loss

    return dice_loss


def ssim_loss(preds, labels):
    n, c, h, w = preds.shape
    pixel_total_num = h * w
    C = 0.000001
    # C1 = 0.01**2
    # C2 = 0.03**2
    ss_loss = 0.0
    for i in range(n):
        pred_mean = torch.mean(preds[i, :, :, :])
        pred_var = torch.var(preds[i, :, :, :])
        label_mean = torch.mean(labels[i, :, :, :])
        label_var = torch.var(labels[i, :, :, :])
        pred_label_var = (
            torch.abs(preds[i, :, :, :] - pred_mean) * torch.abs(labels[i, :, :, :] - label_mean)
        ).sum() / (pixel_total_num - 1)

        # temp_loss = ((torch.square(pred_mean) + torch.square(label_mean)) * (pred_var + label_var) + C) / (
        #         (2 * pred_mean * label_mean) * (2 * pred_label_var) + C)
        temp_loss = (pred_var * label_var + C) / (pred_label_var + C)
        ss_loss = ss_loss + temp_loss

    return ss_loss


def iou_loss(preds, labels):
    iou_loss = 0.0
    n = preds.shape[0]
    C = 0.000001
    for i in range(n):
        Iand = torch.sum(preds[i, :, :, :] * labels[i, :, :, :])
        Ior = torch.sum(preds[i, :, :, :]) + torch.sum(labels[i, :, :, :]) - Iand

        # temp_loss = -torch.log((Iand + C) / (Ior + C))
        temp_loss = Iand / Ior
        iou_loss = iou_loss + (1 - temp_loss)

    return iou_loss


def tversky_loss(preds, labels):
    tversky_loss = 0.0
    beta = 0.7
    alpha = 1.0 - beta
    C = 0.000001
    n = preds.shape[0]
    for i in range(n):
        tp = torch.sum(preds[i, :, :, :] * labels[i, :, :, :])
        fp = torch.sum(preds[i, :, :, :] * (1 - labels[i, :, :, :]))
        fn = torch.sum((1 - preds[i, :, :, :]) * labels[i, :, :, :])
        temp_loss = -torch.log((tp + C) / (tp + alpha * fp + beta * fn + C))

        tversky_loss = tversky_loss + temp_loss

    return tversky_loss


def focal_tversky_loss_plus_plus(preds, labels, gamma: float = 2, beta: float = 0.7, delta: float = 0.75):
    focal_tversky_loss = 0.0
    epsilon = 1e-7
    n = preds.shape[0]
    for i in range(n):
        tp = torch.sum(preds[i, :, :, :] * labels[i, :, :, :])
        fp = torch.sum((preds[i, :, :, :] * (1 - labels[i, :, :, :])) ** gamma)
        fn = torch.sum(((1 - preds[i, :, :, :]) * labels[i, :, :, :]) ** gamma)
        tversky = (tp + (1 - beta) * fp + beta * fn + epsilon) / (tp + epsilon)
        temp_loss = torch.pow(tversky, delta)
        if temp_loss.data.item() > 50.0:
            temp_loss = torch.clamp(temp_loss, max=50.0)

        focal_tversky_loss = focal_tversky_loss + temp_loss

    return focal_tversky_loss


def focal_loss(preds, labels, alpha: float = 0.25, gamma: float = 2, reduction: str = "sum"):
    bce_cross_entropy = F.binary_cross_entropy(preds, labels, reduction=reduction)
    pt = torch.exp(-bce_cross_entropy)
    focal_loss = alpha * ((1 - pt) ** gamma) * bce_cross_entropy

    return focal_loss


def normalize_focalloss(preds, labels, gamma=2.0):
    """
    Args:
        y_pred: [N, 1, H, W] or [N, H, W]
        y_true: [N, H, W] with values 0 and 255
        gamma: scalar
    Returns:
        loss: scalar
    """

    bce_loss = F.binary_cross_entropy(preds, labels, reduction="none")
    pt = torch.exp(-bce_loss)
    modulating_factor = (1 - pt).pow(gamma)
    normalizer = bce_loss.sum() / (modulating_factor * bce_loss).sum()
    scale = normalizer * modulating_factor
    losses = (scale * bce_loss).sum() / (labels.sum() + preds.size(0))

    return losses


def grad_loss(preds, labels_float32):
    device = preds.device

    filter_x = (
        torch.tensor([[-1 / 8, 0, 1 / 8], [-2 / 8, 0, 2 / 8], [-1 / 8, 0, 1 / 8]], dtype=torch.float32)
        .view(1, 1, 3, 3)
        .to(device)
    )
    filter_y = (
        torch.tensor([[-1 / 8, -2 / 8, -1 / 8], [0, 0, 0], [1 / 8, 2 / 8, 1 / 8]], dtype=torch.float32)
        .view(1, 1, 3, 3)
        .to(device)
    )

    filter_xy = torch.cat([filter_x, filter_y], dim=0)

    grad_preds = F.conv2d(preds, filter_xy, padding=1)
    grad_labels = F.conv2d(labels_float32, filter_xy, padding=1)

    return F.l1_loss(grad_preds, grad_labels)


class Binary_Adaptive_Region_Specific_TverskyLoss(nn.Module):
    def __init__(self, smooth=1e-7, num_region_per_axis=(16, 16), batch_dice=True, A=0.3, B=0.4):
        super(Binary_Adaptive_Region_Specific_TverskyLoss, self).__init__()
        self.smooth = smooth
        self.batch_dice = batch_dice
        self.pool = nn.AdaptiveAvgPool2d(num_region_per_axis)
        self.A = A
        self.B = B

    def forward(self, preds, labels):
        # preds: [batch_size, 1, H, W] - 预测的边缘概率
        # labels: [batch_size, H, W] - 二值化的真实边缘标签
        tp = preds * labels
        fp = preds * (1 - labels)
        fn = (1 - preds) * labels

        region_tp = self.pool(tp)
        region_fp = self.pool(fp)
        region_fn = self.pool(fn)

        if self.batch_dice:
            region_tp = region_tp.sum(0)
            region_fp = region_fp.sum(0)
            region_fn = region_fn.sum(0)

        alpha = self.A + self.B * (region_fp + self.smooth) / (region_fp + region_fn + self.smooth)
        beta = self.A + self.B * (region_fn + self.smooth) / (region_fp + region_fn + self.smooth)

        # 计算区域特定的 Tversky 指数
        region_tversky = (region_tp + self.smooth) / (region_tp + alpha * region_fp + beta * region_fn + self.smooth)
        region_tversky = 1 - region_tversky

        if self.batch_dice:
            # [C,H,W]
            region_tversky = region_tversky.sum(list(range(1, len(preds.shape) - 1)))

        return region_tversky.mean()


if __name__ == "__main__":
    loss = Binary_Adaptive_Region_Specific_TverskyLoss()
    preds = torch.randn([8, 1, 480, 320])
    labels = torch.randn([8, 1, 480, 320])
    out = loss(preds, labels)
    print(out)

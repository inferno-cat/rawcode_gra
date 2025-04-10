import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.ops import deform_conv2d
import numpy as np


class Channel_Max_Pooling(nn.Module):
    def __init__(self):
        super(Channel_Max_Pooling, self).__init__()
        self.max_pooling = nn.AdaptiveMaxPool2d((1, 2, 3, 4))

    def forward(self, x):
        print("Input_Shape:", x.shape)  # (batch_size, chs, h, w)
        x = x.transpose(1, 3)  # (batch_size, w, h, chs)
        print("Transpose_Shape:", x.shape)
        x = self.max_pooling(x)
        print("Transpose_MaxPooling_Shape:", x.shape)
        out = x.transpose(1, 3)
        print("Output_Shape:", out.shape)
        return out


# x = torch.randn([3, 64, 40, 40])
# n, c, h, w = x.size()
# x = x.transpose(1, 3)  # n,w,h,c
# pool = nn.AdaptiveMaxPool2d((h, 1))
# y = pool(x)
# y = y.transpose(1, 3)
# print(y.size())
#
# a = torch.randn([1, 3, 40, 40])
# b = torch.randn([1, 1, 40, 40])
# print((a * b).size())

# from model import DeFFN, EdgeNeXt
# from loss import focal_loss
#
# preds = torch.randint(0, 2, size=(3, 2, 5, 5))
# labels = torch.randint(0, 2, size=(3, 2, 5, 5))
# print(labels)
# print(preds)
# loss_2 = focal_loss(preds.float(), labels.float(), reduction="mean")
# loss_3 = focal_loss(preds.float(), labels.float(), reduction="sum")
# print(loss_2)
# print(loss_3)


# def focal(preds, labels, gamma=2, reduction="sum"):
#     weights = (1 - preds) ** gamma
#     loss = F.binary_cross_entropy(preds, labels, reduction=reduction)
#
#     return loss
#
#
# loss = focal(preds.float(), labels.float())
# loss_m = focal(preds.float(), labels.float(), reduction="mean")
# print(loss)
# print(loss_m)

# back_dice = (1 - dice_class[:, 0]) * torch.pow(1 - dice_class[:, 0], -0.75)
# fore_dice = (1 - dice_class[:, 1]) * torch.pow(1 - dice_class[:, 1], -0.75)
# print(dice_class, dice_class.shape)

# from model import MBEdgeNet
# model = MBEdgeNet()
# for key,value in model.named_modules():
#     if key.find("laplacian") == -1:
#         if isinstance(value, nn.Conv2d):
#             torch.nn.init.kaiming_normal_(value.weight, mode='fan_out', nonlinearity='relu')
#             print(key, value)
# input=torch.randn(1,256,256,256)
# model=nn.Sequential(*list(models.resnet18(pretrained=False).layer4))
# out=model(input)
# print(out.shape)
# print(models.resnet18(pretrained=False))


model = models.resnext50_32x4d()
print(model)

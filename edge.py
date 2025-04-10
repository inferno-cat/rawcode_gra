import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2d(nn.Module):
    def __init__(
        self, pdc_func, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False
    ):
        super(Conv2d, self).__init__()
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()
        self.pdc_func = createConvFunc(pdc_func)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return self.pdc_func(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


## cd, ad, rd convolutions
## theta could be used to control the vanilla conv components
## theta = 0 reduces the function to vanilla conv, theta = 1 reduces the fucntion to pure pdc (used in the paper)
def createConvFunc(op_type):
    assert op_type in [
        "cv",
        "cd",
        "ad",
        "rd",
        "vertical",
        "horizontal",
        "cross",
        "diagonal",
    ], "unknown op type: %s" % str(op_type)
    if op_type == "cv":
        return F.conv2d

    # assert theta > 0 and theta <= 1.0, "theta should be within (0, 1]"

    if op_type == "cd":

        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], "dilation for cd_conv should be in 1 or 2"
            assert weights.size(2) == 3 and weights.size(3) == 3, "kernel size for cd_conv should be 3x3"
            assert padding == dilation, "padding for cd_conv set wrong"

            weights_c = weights.sum(dim=[2, 3], keepdim=True)
            yc = F.conv2d(x, weights_c, stride=stride, padding=0, groups=groups)
            y = F.conv2d(x, weights, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y - yc

        return func
    elif op_type == "ad":

        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], "dilation for ad_conv should be in 1 or 2"
            assert weights.size(2) == 3 and weights.size(3) == 3, "kernel size for ad_conv should be 3x3"
            assert padding == dilation, "padding for ad_conv set wrong"

            shape = weights.shape
            weights = weights.view(shape[0], shape[1], -1)
            weights_conv = (weights - weights[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]).view(shape)  # clock-wise
            y = F.conv2d(x, weights_conv, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y

        return func
    elif op_type == "rd":

        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], "dilation for rd_conv should be in 1 or 2"
            assert weights.size(2) == 3 and weights.size(3) == 3, "kernel size for rd_conv should be 3x3"
            padding = 2 * dilation

            shape = weights.shape
            if weights.is_cuda:
                buffer = torch.cuda.FloatTensor(shape[0], shape[1], 5 * 5).fill_(0)
            else:
                buffer = torch.zeros(shape[0], shape[1], 5 * 5)
            weights = weights.view(shape[0], shape[1], -1)
            buffer[:, :, [0, 2, 4, 10, 14, 20, 22, 24]] = weights[:, :, 1:]
            buffer[:, :, [6, 7, 8, 11, 13, 16, 17, 18]] = -weights[:, :, 1:]
            buffer[:, :, 12] = weights[:, :, 0]
            buffer = buffer.view(shape[0], shape[1], 5, 5)
            y = F.conv2d(x, buffer, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y

        return func

    elif op_type == "vertical":

        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], "dilation for ad_conv should be in 1 or 2"
            assert weights.size(2) == 3 and weights.size(3) == 3, "kernel size for ad_conv should be 3x3"
            assert padding == dilation, "padding for ad_conv set wrong"

            shape = weights.shape
            # 0 1 2
            # 3 4 5   ==>  0 1 2 3 4 5 6 7 8
            # 6 7 8
            weights = weights.view(shape[0], shape[1], -1)
            weights_conv = weights[:, :, [0, 1, 2, 3, 4, 5, 6, 7, 8]] - weights[:, :, [6, 7, 8, 0, 1, 2, 3, 4, 5]]
            weights_conv = weights_conv.view(shape)
            y = F.conv2d(x, weights_conv, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y

        return func
    elif op_type == "horizontal":

        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], "dilation for ad_conv should be in 1 or 2"
            assert weights.size(2) == 3 and weights.size(3) == 3, "kernel size for ad_conv should be 3x3"
            assert padding == dilation, "padding for ad_conv set wrong"

            shape = weights.shape
            # 0 1 2
            # 3 4 5   ==>  0 1 2 3 4 5 6 7 8
            # 6 7 8
            weights = weights.view(shape[0], shape[1], -1)
            weights_conv = weights[:, :, [0, 1, 2, 3, 4, 5, 6, 7, 8]] - weights[:, :, [2, 0, 1, 5, 3, 4, 8, 6, 7]]
            weights_conv = weights_conv.view(shape)
            y = F.conv2d(x, weights_conv, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y

        return func
    elif op_type == "cross":

        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], "dilation for ad_conv should be in 1 or 2"
            assert weights.size(2) == 3 and weights.size(3) == 3, "kernel size for ad_conv should be 3x3"
            assert padding == dilation, "padding for ad_conv set wrong"

            shape = weights.shape
            # 0 1 2
            # 3 4 5   ==>  0 1 2 3 4 5 6 7 8
            # 6 7 8
            weights = weights.view(shape[0], shape[1], -1)
            weights_conv = weights.clone()
            weights_conv[:, :, [1, 3, 5, 7]] = weights_conv[:, :, [1, 3, 5, 7]] - weights_conv[:, :, [7, 5, 4, 4]]
            weights_conv[:, :, [4]] = 2 * weights_conv[:, :, [4]] - weights_conv[:, :, [3]] - weights_conv[:, :, [1]]
            weights_conv[:, :, [0, 2, 6, 8]] = 0
            weights_conv = weights_conv.view(shape)
            y = F.conv2d(x, weights_conv, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y

        return func
    elif op_type == "diagonal":

        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], "dilation for ad_conv should be in 1 or 2"
            assert weights.size(2) == 3 and weights.size(3) == 3, "kernel size for ad_conv should be 3x3"
            assert padding == dilation, "padding for ad_conv set wrong"

            shape = weights.shape
            # 0 1 2
            # 3 4 5   ==>  0 1 2 3 4 5 6 7 8
            # 6 7 8
            weights = weights.view(shape[0], shape[1], -1)
            weights_conv = weights.clone()
            weights_conv[:, :, [0, 2, 6, 8]] = weights_conv[:, :, [0, 2, 6, 8]] - weights_conv[:, :, [8, 6, 4, 4]]
            weights_conv[:, :, [4]] = 2 * weights_conv[:, :, [4]] - weights_conv[:, :, [0]] - weights_conv[:, :, [2]]
            weights_conv[:, :, [1, 3, 5, 7]] = 0
            weights_conv = weights_conv.view(shape)
            y = F.conv2d(x, weights_conv, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y

        return func

    else:
        print("impossible to be here unless you force that")
        return None


class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False):
        super(Conv2d_cd, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight = conv_weight.view(conv_shape[0], conv_shape[1], -1)
        if conv_weight.is_cuda:
            conv_weight_cd = torch.cuda.FloatTensor(conv_shape[0], conv_shape[1], 3 * 3).fill_(0)
        else:
            conv_weight_cd = torch.zeros(conv_shape[0], conv_shape[1], 3 * 3)
        conv_weight_cd[:, :, :] = conv_weight[:, :, :]
        conv_weight_cd[:, :, 4] = conv_weight[:, :, 4] - conv_weight[:, :, :].sum(2)
        conv_weight_cd = conv_weight_cd.view(conv_shape)

        return conv_weight_cd, self.conv.bias


class Conv2d_ad(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False):
        super(Conv2d_ad, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight_ad = conv_weight.view(conv_shape[0], conv_shape[1], -1).clone()
        conv_weight_ad = conv_weight_ad - conv_weight_ad[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]
        conv_weight_ad = conv_weight_ad.view(conv_shape)

        return conv_weight_ad, self.conv.bias


class Conv2d_rd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False):
        super(Conv2d_rd, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight = conv_weight.view(conv_shape[0], conv_shape[1], -1)
        if conv_weight.is_cuda:
            buffer = torch.cuda.FloatTensor(conv_shape[0], conv_shape[1], 5 * 5).fill_(0)
        else:
            buffer = torch.zeros(conv_shape[0], conv_shape[1], 5 * 5)
        buffer[:, :, [0, 2, 4, 10, 14, 20, 22, 24]] = conv_weight[:, :, 1:]
        buffer[:, :, [6, 7, 8, 11, 13, 16, 17, 18]] = -conv_weight[:, :, 1:]
        buffer[:, :, 12] = conv_weight[:, :, 0]
        conv_weight_rd = buffer.view(conv_shape[0], conv_shape[1], 5, 5)

        return conv_weight_rd, self.conv.bias


class Conv2d_v(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False):
        super(Conv2d_v, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight_v = conv_weight.view(conv_shape[0], conv_shape[1], -1).clone()
        # if conv_weight.is_cuda:
        #     conv_weight_v = torch.cuda.FloatTensor(
        #         conv_shape[0], conv_shape[1], 3 * 3
        #     ).fill_(0)
        # else:
        #     conv_weight_v = torch.zeros(conv_shape[0], conv_shape[1], 3 * 3)
        conv_weight_v = conv_weight_v - conv_weight_v[:, :, [6, 7, 8, 0, 1, 2, 3, 4, 5]]
        conv_weight_v = conv_weight_v.view(conv_shape)
        return conv_weight_v, self.conv.bias


class Conv2d_h(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False):
        super(Conv2d_h, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight_h = conv_weight.view(conv_shape[0], conv_shape[1], -1).clone()
        # if conv_weight.is_cuda:
        #     conv_weight_h = torch.cuda.FloatTensor(
        #         conv_shape[0], conv_shape[1], 3 * 3
        #     ).fill_(0)
        # else:
        #     conv_weight_h = torch.zeros(conv_shape[0], conv_shape[1], 3 * 3)
        conv_weight_h = conv_weight_h - conv_weight_h[:, :, [2, 0, 1, 5, 3, 4, 8, 6, 7]]
        conv_weight_h = conv_weight_h.view(conv_shape)
        return conv_weight_h, self.conv.bias


class Conv2d_c(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False):
        super(Conv2d_c, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight_c = conv_weight.view(conv_shape[0], conv_shape[1], -1).clone()
        # if conv_weight.is_cuda:
        #     conv_weight_c = torch.cuda.FloatTensor(
        #         conv_shape[0], conv_shape[1], 3 * 3
        #     ).fill_(0)
        # else:
        #     conv_weight_c = torch.zeros(conv_shape[0], conv_shape[1], 3 * 3)
        conv_weight_c[:, :, [1, 3, 5, 7]] = conv_weight_c[:, :, [1, 3, 5, 7]] - conv_weight_c[:, :, [7, 5, 4, 4]]
        conv_weight_c[:, :, [4]] = 2 * conv_weight_c[:, :, [4]] - conv_weight_c[:, :, [3]] - conv_weight_c[:, :, [1]]
        conv_weight_c[:, :, [0, 2, 6, 8]] = 0
        conv_weight_c = conv_weight_c.view(conv_shape)
        return conv_weight_c, self.conv.bias


class Conv2d_d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False):
        super(Conv2d_d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight_d = conv_weight.view(conv_shape[0], conv_shape[1], -1).clone()
        # if conv_weight.is_cuda:
        #     conv_weight_d = torch.cuda.FloatTensor(
        #         conv_shape[0], conv_shape[1], 3 * 3
        #     ).fill_(0)
        # else:
        #     conv_weight_d = torch.zeros(conv_shape[0], conv_shape[1], 3 * 3)
        conv_weight_d[:, :, [0, 2, 4, 6]] = conv_weight_d[:, :, [0, 2, 4, 6]] - conv_weight_d[:, :, [8, 6, 4, 4]]
        conv_weight_d[:, :, [4]] = 2 * conv_weight_d[:, :, [4]] - conv_weight_d[:, :, [0]] - conv_weight_d[:, :, [2]]
        conv_weight_d[:, :, [1, 3, 5, 7]] = 0
        conv_weight_d = conv_weight_d.view(conv_shape)
        return conv_weight_d, self.conv.bias


class ConvFac(nn.Module):
    def __init__(
        self, op_type, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False
    ):
        super(ConvFac, self).__init__()
        assert op_type in [Conv2d_v, Conv2d_h, Conv2d_c, Conv2d_d, Conv2d_cd, Conv2d_ad, Conv2d_rd]
        self.op_type = op_type(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.standard_conv = None

    def forward(self, x):
        if self.standard_conv is None:
            w, b = self.op_type.get_weight()
            res = F.conv2d(x, weight=w, bias=b, stride=self.stride, padding=self.padding, groups=self.groups)
        else:
            res = self.standard_conv(x)

        return res

    def convert_to_standard_conv(self):
        w, b = self.op_type.get_weight()
        self.standard_conv = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            w.size(2),
            stride=self.stride,
            padding=self.padding,
            groups=self.groups,
            bias=b is not None,
        )
        self.standard_conv.weight.data = w
        if b is not None:
            self.standard_conv.bias.data = b

    def load_state_dict(self, state_dict, strict=True):
        if 'standard_conv.weight' in state_dict:
            if self.standard_conv is None:
                self.convert_to_standard_conv()
            self.standard_conv.load_state_dict(state_dict, strict)
        else:
            super().load_state_dict(state_dict, strict)


class CPDCBlock(nn.Module):
    def __init__(self, in_channels):
        super(CPDCBlock, self).__init__()
        # self.conv_cd = nn.Sequential(
        #     ConvFac(Conv2d_cd, in_channels, in_channels // 4, 3, 1, 1, groups=in_channels // 4),
        #     nn.BatchNorm2d(in_channels // 4),
        #     nn.ReLU(True),
        # )
        #
        # self.conv_ad = nn.Sequential(
        #     ConvFac(Conv2d_ad, in_channels, in_channels // 4, 3, 1, 1, groups=in_channels // 4),
        #     nn.BatchNorm2d(in_channels // 4),
        #     nn.ReLU(True),
        # )

        # self.conv_rd = nn.Sequential(
        #     ConvFac(Conv2d_rd, in_channels, in_channels // 4, 3, 1, 2, groups=in_channels // 4),
        #     nn.BatchNorm2d(in_channels // 4),
        #     nn.ReLU(True),
        # )

        self.conv_d = nn.Sequential(
            ConvFac(Conv2d_d, in_channels, in_channels // 4, 3, 1, 1, groups=in_channels // 4),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(True),
        )

        self.conv_v = nn.Sequential(
            ConvFac(Conv2d_v, in_channels, in_channels // 4, 3, 1, 1, groups=in_channels // 4),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(True),
        )

        self.conv_h = nn.Sequential(
            ConvFac(Conv2d_h, in_channels, in_channels // 4, 3, 1, 1, groups=in_channels // 4),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(True),
        )

        self.conv_c = nn.Sequential(
            ConvFac(Conv2d_c, in_channels, in_channels // 4, 3, 1, 1, groups=in_channels // 4),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(True),
        )

        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
        )

        self.conv1x1 = nn.Conv2d(in_channels, in_channels, 1, 1, 0)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, x):
        residual = x

        x_d = self.conv_d(x)
        x_v = self.conv_v(x)
        x_h = self.conv_h(x)
        x_c = self.conv_c(x)

        # x_d = self.conv_cd(x)
        # x_v = self.conv_ad(x)
        # x_h = self.conv_d(x)
        # x_c = self.conv_c(x)

        x = torch.cat([x_d, x_v, x_h, x_c], dim=1)

        x = self.conv3x3(x)
        x = self.conv1x1(x)

        x = x + residual

        return x

    def convert_to_standard_conv(self):
        for module in self.modules():
            if isinstance(module, ConvFac):
                module.convert_to_standard_conv()


class PlainBlock(nn.Module):
    def __init__(self, in_channels):
        super(PlainBlock, self).__init__()
        self.conv_d = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 3, 1, 1, groups=in_channels // 4),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(True),
        )

        self.conv_v = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 3, 1, 1, groups=in_channels // 4),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(True),
        )

        self.conv_h = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 3, 1, 1, groups=in_channels // 4),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(True),
        )

        self.conv_c = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 3, 1, 1, groups=in_channels // 4),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(True),
        )

        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
        )

        self.conv1x1 = nn.Conv2d(in_channels, in_channels, 1, 1, 0)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, x):
        residual = x

        x_d = self.conv_d(x)
        x_v = self.conv_v(x)
        x_h = self.conv_h(x)
        x_c = self.conv_c(x)

        x = torch.cat([x_d, x_v, x_h, x_c], dim=1)

        x = self.conv3x3(x)
        x = self.conv1x1(x)

        x = x + residual

        return x


class EdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1, bias=False):
        super(EdgeConv, self).__init__()
        self.bias = bias
        self.groups = groups
        self.conv_v = ConvFac(Conv2d_v, in_channels, out_channels, 3, groups=groups, bias=bias)
        self.conv_h = ConvFac(Conv2d_h, in_channels, out_channels, 3, groups=groups, bias=bias)
        self.conv_c = ConvFac(Conv2d_h, in_channels, out_channels, 3, groups=groups, bias=bias)
        self.conv_d = ConvFac(Conv2d_h, in_channels, out_channels, 3, groups=groups, bias=bias)
        self.conv_p = nn.Conv2d(in_channels, out_channels, 3, groups=groups, bias=bias)

    def forward(self, x):
        w_v, b_v = self.conv_v.get_weight()
        w_h, b_h = self.conv_h.get_weight()
        w_c, b_c = self.conv_c.get_weight()
        w_d, b_d = self.conv_d.get_weight()
        w_p, b_p = self.conv_p.weight, self.conv_p.bias

        w = w_v + w_h + w_c + w_d + w_p

        if self.bias:
            b = b_v + b_h + b_c + b_d + b_p
        else:
            b = None

        res = F.conv2d(x, weight=w, bias=b, stride=1, padding=1, groups=self.groups)

        return res


if __name__ == "__main__":
    # weights = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.float32)
    # weights_conv = weights.clone()
    # theta = 1.0
    # weights_conv[[1, 3]] = weights[[1, 3]] - theta * weights[[4]]
    # weights_conv[[5, 7]] = weights[[4]] - theta * weights[[5, 7]]
    # weights_conv[[0, 2, 6, 8]] = 1 - theta
    # print(weights_conv.view(3, 3))
    conv = CPDCBlock(16)
    # conv = ConvFactor(Conv2d, 3, 3, bias=False)
    # print(conv)
    x = torch.randn(1, 16, 480, 320)
    out = conv(x)
    print(out.shape)

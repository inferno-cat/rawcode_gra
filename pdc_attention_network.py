import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from edge import Conv2d, EdgeConv, CPDCBlock, PlainBlock
from timm.models.layers import trunc_normal_, DropPath


class BaseConv(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, activation=None, use_bn=False
    ):
        super(BaseConv, self).__init__()
        self.use_bn = use_bn
        self.activation = activation
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2, dilation, groups)
        self.conv.weight.data.normal_(0, 0.01)
        self.conv.bias.data.zero_()
        self.bn = nn.BatchNorm2d(out_channels)
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()

    def forward(self, input):
        input = self.conv(input)
        if self.use_bn:
            input = self.bn(input)
        if self.activation:
            input = self.activation(input)

        return input


class LayerNorm(nn.Module):
    """LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class GRN(nn.Module):
    """GRN (Global Response Normalization) layer"""

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class ConvNeXtV2_Block(nn.Module):
    """ConvNeXtV2 Block.

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """

    def __init__(self, dim, drop_path=0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class LaplacianConv2d(nn.Module):
    def __init__(self, dim):
        super(LaplacianConv2d, self).__init__()
        self.laplacian = self._get_kernel(dim, dim)

    def _get_kernel(self, in_channels, out_channels):
        filter_laplacian = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]]).astype(np.float32)

        filter_laplacian = filter_laplacian.reshape((1, 1, 3, 3))
        filter_laplacian = np.repeat(filter_laplacian, in_channels, axis=1)
        filter_laplacian = np.repeat(filter_laplacian, out_channels, axis=0)
        filter_laplacian = torch.from_numpy(filter_laplacian)
        filter_laplacian = nn.Parameter(filter_laplacian, requires_grad=False)

        conv_laplacian = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        conv_laplacian.weight = filter_laplacian

        return conv_laplacian

    def forward(self, x):
        return self.laplacian(x)


class SobelConv2d(nn.Module):
    def __init__(self, dim):
        super(SobelConv2d, self).__init__()
        self.sobel_x, self.sobel_y = self._get_kernel(dim, dim)

    def _get_kernel(self, in_channels, out_channels):
        sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).astype(np.float32)

        filter_sobel_x = sobel_x.reshape((1, 1, 3, 3))
        filter_sobel_x = np.repeat(filter_sobel_x, in_channels, axis=1)
        filter_sobel_x = np.repeat(filter_sobel_x, out_channels, axis=0)
        filter_sobel_x = torch.from_numpy(filter_sobel_x)
        filter_sobel_x = nn.Parameter(filter_sobel_x, requires_grad=False)

        conv_sobel_x = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        conv_sobel_x.weight = filter_sobel_x

        sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).astype(np.float32)

        filter_sobel_y = sobel_y.reshape((1, 1, 3, 3))
        filter_sobel_y = np.repeat(filter_sobel_y, in_channels, axis=1)
        filter_sobel_y = np.repeat(filter_sobel_y, out_channels, axis=0)
        filter_sobel_y = torch.from_numpy(filter_sobel_y)
        filter_sobel_y = nn.Parameter(filter_sobel_y, requires_grad=False)

        conv_sobel_y = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        conv_sobel_y.weight = filter_sobel_y

        return conv_sobel_x, conv_sobel_y

    def forward(self, x):
        return self.sobel_x(x) + self.sobel_y(x)


class Squeeze_and_Excitation_Module(nn.Module):
    def __init__(self, channel, reduction_ratio=16):
        super(Squeeze_and_Excitation_Module, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction_ratio, channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class MultiScaleContextModule(nn.Module):
    def __init__(self, dim):
        super(MultiScaleContextModule, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, dim // 4, 3, 1, 1, dilation=1),
            nn.ReLU(inplace=True),
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, dim // 4, 3, 1, 2, dilation=2),
            nn.ReLU(inplace=True),
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, dim // 4, 3, 1, 3, dilation=3),
            nn.ReLU(inplace=True),
        )

        self.branch4 = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, dim // 4, 3, 1, 4, dilation=4),
            nn.ReLU(inplace=True),
        )

        self.conv1x1 = nn.Conv2d(dim, dim, 1, 1, 0)

        self.attn = Squeeze_and_Excitation_Module(dim)

    def forward(self, x):
        residual = x

        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        o = torch.cat([branch1, branch2, branch3, branch4], dim=1)
        o = self.conv1x1(o)
        o = self.attn(o)
        o += residual

        return o


class MSBlock(nn.Module):
    def __init__(self, c_in, rate=4):
        super(MSBlock, self).__init__()
        c_out = c_in
        self.rate = rate

        self.conv = nn.Conv2d(c_in, 32, 3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        dilation = self.rate * 1 if self.rate >= 1 else 1
        self.conv1 = nn.Conv2d(32, 32, 3, stride=1, dilation=dilation, padding=dilation)
        self.relu1 = nn.ReLU(inplace=True)
        dilation = self.rate * 2 if self.rate >= 1 else 1
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1, dilation=dilation, padding=dilation)
        self.relu2 = nn.ReLU(inplace=True)
        dilation = self.rate * 3 if self.rate >= 1 else 1
        self.conv3 = nn.Conv2d(32, 32, 3, stride=1, dilation=dilation, padding=dilation)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv1x1 = nn.Conv2d(32, c_in, 1, 1, 0)

    def forward(self, x):
        o = self.relu(self.conv(x))
        o1 = self.relu1(self.conv1(o))
        o2 = self.relu2(self.conv2(o))
        o3 = self.relu3(self.conv3(o))
        out = self.conv1x1(o + o1 + o2 + o3)
        return out


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()
        self.down = nn.Conv2d(in_channels, in_channels, 3, 2, 1, bias=False)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)

    def forward(self, x):
        x = self.down(x)
        x = self.conv1x1(x)

        return x


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(UpSample, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        self.up = nn.Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=True)

    def forward(self, x):
        o = self.conv(x)
        o = self.up(o)

        return o


class Decoder(nn.Module):
    def __init__(self, in_channels):
        super(Decoder, self).__init__()

        self.conv0 = BaseConv(in_channels, in_channels, 3, 1, activation=nn.ReLU(inplace=True), use_bn=True)

        self.conv1 = BaseConv(in_channels, in_channels // 2, 1, 1, activation=nn.ReLU(inplace=True), use_bn=True)

        self.conv2 = BaseConv(in_channels // 2, in_channels // 2, 3, 1, activation=nn.ReLU(inplace=True), use_bn=True)

        self.conv3 = BaseConv(in_channels // 2, in_channels, 1, 1, activation=None, use_bn=True)

        self.conv4 = BaseConv(in_channels, in_channels, 3, 1, use_bn=True)

    def forward(self, x):
        residual = x

        x0 = self.conv0(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = F.relu(x + x0)

        x = self.conv4(x)
        x = x + residual

        return F.relu(x)


class PDCNet(nn.Module):
    def __init__(self, base_dim=16):
        super(PDCNet, self).__init__()
        # self.block = PlainBlock
        self.block = CPDCBlock
        self.in_channels = [base_dim, base_dim * 2, base_dim * 4, base_dim * 4]
        self.stem_conv = nn.Sequential(
            nn.Conv2d(3, self.in_channels[0], 3, 1, 1, bias=False),
            nn.Conv2d(self.in_channels[0], self.in_channels[0], 3, 1, 1, bias=False),
        )

        self.stage1 = self._make_layer(self.block, self.in_channels[0], 4)
        self.stage2 = self._make_layer(self.block, self.in_channels[1], 4)
        self.stage3 = self._make_layer(self.block, self.in_channels[2], 4)
        self.stage4 = self._make_layer(self.block, self.in_channels[3], 4)

        self.down2 = DownSample(self.in_channels[0], self.in_channels[1])
        self.down3 = DownSample(self.in_channels[1], self.in_channels[2])
        self.down4 = DownSample(self.in_channels[2], self.in_channels[3])

        # self.mscm4 = MultiScaleContextModule(self.in_channels[3])
        # self.mscm3 = MultiScaleContextModule(self.in_channels[2])
        # self.mscm2 = MultiScaleContextModule(self.in_channels[1])
        # self.mscm1 = MultiScaleContextModule(self.in_channels[0])

        self.mscm4 = MSBlock(self.in_channels[3])
        self.mscm3 = MSBlock(self.in_channels[2])
        self.mscm2 = MSBlock(self.in_channels[1])
        self.mscm1 = MSBlock(self.in_channels[0])

        self.de3 = Decoder(self.in_channels[2])
        self.de2 = Decoder(self.in_channels[1])
        self.de1 = Decoder(self.in_channels[0])
        # self.de3 = nn.Conv2d(self.in_channels[2], self.in_channels[2], 1, 1, 0, bias=False)
        # self.de2 = nn.Conv2d(self.in_channels[1], self.in_channels[1], 1, 1, 0, bias=False)
        # self.de1 = nn.Conv2d(self.in_channels[0], self.in_channels[0], 1, 1, 0, bias=False)

        self.up4 = UpSample(self.in_channels[3], self.in_channels[2])
        self.up3 = UpSample(self.in_channels[2], self.in_channels[1])
        self.up2 = UpSample(self.in_channels[1], self.in_channels[0])

        self.convnext = nn.Sequential(
            BaseConv(3, self.in_channels[0], 3, 1, activation=nn.ReLU(inplace=True), use_bn=True),
            ConvNeXtV2_Block(self.in_channels[0]),
        )

        # self.edge3 = BaseConv(base_dim * 4, 1, 1, 1)
        # self.edge2 = BaseConv(base_dim * 2, 1, 1, 1)
        # self.edge1 = BaseConv(base_dim, 1, 1, 1)

        self.output_layer = nn.Sequential(
            BaseConv(2 * self.in_channels[0], self.in_channels[0], 1, 1, activation=nn.ReLU(inplace=True)),
            BaseConv(self.in_channels[0], 1, 3, 1),
        )

    def _make_layer(self, block, dim, block_nums):
        layers = []

        for i in range(0, block_nums):
            layers.append(block(dim))

        return nn.Sequential(*layers)

    def forward(self, x):
        convnext = self.convnext(x)

        conv_stem = self.stem_conv(x)

        conv1 = self.stage1(conv_stem)  # C
        conv2 = self.stage2(self.down2(conv1))  # 2C
        conv3 = self.stage3(self.down3(conv2))  # 4C
        conv4 = self.stage4(self.down4(conv3))  # 4C

        mscm4 = self.mscm4(conv4)
        mscm4_up = self.up4(mscm4)  # 4C

        mscm3 = self.mscm3(conv3)
        de3 = self.de3(mscm3 + mscm4_up)
        de3_up = self.up3(de3)  # 2C

        mscm2 = self.mscm2(conv2)
        de2 = self.de2(mscm2 + de3_up)
        de2_up = self.up2(de2)  # C

        mscm1 = self.mscm1(conv1)
        de1 = self.de1(mscm1 + de2_up)

        output = self.output_layer(torch.cat([de1, convnext], dim=1))

        return torch.sigmoid(output)

    def convert_to_standard_conv(self):
        for module in self.modules():
            if isinstance(module, CPDCBlock):
                module.convert_to_standard_conv()

    def load_state_dict(self, state_dict, strict=False):
        model_state_dict = self.state_dict()
        for name, param in state_dict.items():
            if name in model_state_dict:
                model_state_dict[name].co_(param)
            elif name.replace('standard_conv', 'op_type') in model_state_dict:
                # 处理已转换的卷积
                new_name = name.replace('standard_conv', 'op_type')
                model_state_dict[new_name].copy_(param)
        super().load_state_dict(model_state_dict, strict=False)
        self.convert_to_standard_conv()


if __name__ == "__main__":
    net = PDCNet(16)
    x = torch.randn(1, 3, 480, 320)
    y = net(x)
    print(y.shape)

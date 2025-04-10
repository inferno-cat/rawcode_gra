import torch
import torch.nn as nn
from backbone.EfficientNetV2 import EfficientNetV2
import torch.nn.functional as F
from edge import ConvFac, Conv2d_h, Conv2d_v


class DetailEnhancementModule(nn.Module):
    def __init__(self, in_channels):
        super(DetailEnhancementModule, self).__init__()
        self.conv_x = ConvFac(Conv2d_h, in_channels, in_channels // 2, 3, 1, padding=1)
        self.conv_y = ConvFac(Conv2d_v, in_channels, in_channels // 2, 3, 1, padding=1)
        self.merge = nn.Conv2d(in_channels, in_channels, 1, 1, padding=0)
        self.conv_g = nn.Conv2d(in_channels, in_channels, 7, 1, padding=3)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv_x = self.conv_x(x)
        conv_y = self.conv_y(x)
        conv_merge = torch.sigmoid(self.merge(torch.cat([conv_x, conv_y], dim=1)))

        conv_g = self.relu(self.conv_g(x))

        conv_filter = conv_merge * conv_g

        return conv_filter


class LayerNorm4D(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.permute(0, 2, 3, 1).contiguous().view(-1, c)
        x = self.norm(x)
        x = x.view(b, h, w, c).permute(0, 3, 1, 2)
        return x


class upsample(nn.Module):
    def __init__(self, scale_factor=2, in_channels=18, out_channels=12):
        super(upsample, self).__init__()
        self.up = nn.Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=True)
        self.conv = nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels)
        self.down_channels = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        x = self.down_channels(x)

        return x


class LocalMixer(nn.Module):
    def __init__(self, dim):
        super(LocalMixer, self).__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        # self.ln = LayerNorm4D(dim)
        # self.act = nn.GELU()
        self.bn = nn.BatchNorm2d(dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class GlobalMixer(nn.Module):
    def __init__(self, dim):
        super(GlobalMixer, self).__init__()
        self.conv = nn.Conv2d(dim, dim, 7, 1, 3, groups=dim)
        # self.ln = LayerNorm4D(dim)
        # self.act = nn.GELU()
        self.bn = nn.BatchNorm2d(dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = x + residual
        return x


class MixerDecoder(nn.Module):
    def __init__(self, dim):
        super(MixerDecoder, self).__init__()
        self.local_mixer = LocalMixer(dim)
        self.global_mixer = GlobalMixer(dim)
        self.conv1x1 = nn.Sequential(nn.Conv2d(dim, dim, 1, 1, 0), nn.BatchNorm2d(dim), nn.ReLU(inplace=True))

    def forward(self, x):
        residual = x
        x = self.local_mixer(x)
        x = self.global_mixer(x)
        x = self.conv1x1(x)
        x = x + residual
        return x


class FSRelation(nn.Module):
    def __init__(self, scene_embedding_channels, in_channels_list, scale_aware_proj=False):
        super(FSRelation, self).__init__()
        self.scale_aware_proj = scale_aware_proj

        if scale_aware_proj:
            self.scene_encoder = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(scene_embedding_channels, c, 1),
                        nn.GroupNorm(32, c),
                        nn.ReLU(True),
                        nn.Conv2d(c, c, 1),
                        nn.GroupNorm(32, c),
                        nn.ReLU(True),
                    )
                    for c in in_channels_list
                ]
            )
            self.project = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(c * 2, c, 1, bias=False), nn.BatchNorm2d(c), nn.ReLU(True), nn.Dropout2d(p=0.1)
                    )
                    for c in in_channels_list
                ]
            )
        else:
            # 2mlp
            max_channels = max(in_channels_list)
            self.scene_encoder = nn.Sequential(
                nn.Conv2d(scene_embedding_channels, max_channels, 1),
                nn.GroupNorm(32, max_channels),
                nn.ReLU(True),
                nn.Conv2d(max_channels, max_channels, 1),
                nn.GroupNorm(32, max_channels),
                nn.ReLU(True),
            )
            self.project = nn.Sequential(
                nn.Conv2d(max_channels * 2, max_channels, 1, bias=False),
                nn.BatchNorm2d(max_channels),
                nn.ReLU(True),
                nn.Dropout2d(p=0.1),
            )

        self.content_encoders = nn.ModuleList()
        self.feature_reencoders = nn.ModuleList()
        for c in in_channels_list:
            self.content_encoders.append(nn.Sequential(nn.Identity()))
            self.feature_reencoders.append(nn.Sequential(nn.Conv2d(c, c, 1), nn.BatchNorm2d(c), nn.ReLU(True)))

        # self.normalizer = nn.Sigmoid()
        self.normalizer = nn.Softmax(dim=1)

    def forward(self, scene_feature, features: list):
        # [N, C, H, W]
        content_feats = [c_en(p_feat) for c_en, p_feat in zip(self.content_encoders, features)]
        if self.scale_aware_proj:
            scene_feats = [op(scene_feature) for op in self.scene_encoder]
            relations = [
                self.normalizer((sf * cf).sum(dim=1, keepdim=True)) for sf, cf in zip(scene_feats, content_feats)
            ]
        else:
            # [N, C, 1, 1]
            scene_feat = self.scene_encoder(scene_feature)
            relations = [self.normalizer((scene_feat * cf).sum(dim=1, keepdim=True)) for cf in content_feats]

        p_feats = [op(p_feat) for op, p_feat in zip(self.feature_reencoders, features)]

        refined_feats = [torch.cat([r * p, o], dim=1) for r, p, o in zip(relations, p_feats, features)]

        if self.scale_aware_proj:
            ffeats = [op(x) for op, x in zip(self.project, refined_feats)]
        else:
            ffeats = [self.project(x) for x in refined_feats]

        return ffeats


class CAM(nn.Module):
    def __init__(self, in_channels):
        super(CAM, self).__init__()
        self.in_channels = in_channels
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.fuse = nn.Conv2d(in_channels * 2, in_channels, 1, 1, 0)

    def forward(self, x):
        avg = self.avg(x)
        max = self.max(x)
        output = self.fuse(torch.cat([avg, max], dim=1))

        return output


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        self.encoder = EfficientNetV2("l", pretrained=True)

        self.dem4 = DetailEnhancementModule(640)
        self.dem3 = DetailEnhancementModule(224)
        self.dem2 = DetailEnhancementModule(96)
        self.dem1 = DetailEnhancementModule(64)

        self.cam = CAM(640)

        self.fsr = FSRelation(scene_embedding_channels=640, in_channels_list=[224, 96, 64], scale_aware_proj=True)

        self.de3 = MixerDecoder(224)
        self.de2 = MixerDecoder(96)
        self.de1 = MixerDecoder(64)

        self.up4 = upsample(2, 640, 224)
        self.up3 = upsample(2, 224, 96)
        self.up2 = upsample(2, 96, 64)

        self.compress3 = nn.Conv2d(224, 1, 1, 1, 0)
        self.compress2 = nn.Conv2d(96, 1, 1, 1, 0)
        self.compress1 = nn.Conv2d(64, 1, 1, 1, 0)

        self.output_layer = nn.Conv2d(3, 1, 1, 1, 0)

    def forward(self, x):
        conv1, conv2, conv3, conv4 = self.encoder.get_features(x)

        dem4 = self.dem4(conv4)  # 256
        dem3 = self.dem3(conv3)  # 96
        dem2 = self.dem2(conv2)  # 64
        dem1 = self.dem1(conv1)  # 32

        dem3_fuse = self.up4(dem4) + dem3  # 96
        dem2_fuse = self.up3(dem3_fuse) + dem2  # 64
        dem1_fuse = self.up2(dem2_fuse) + dem1  # 32

        scene_embedding = self.cam(conv4)
        fsr = self.fsr(scene_embedding, [dem3_fuse, dem2_fuse, dem1_fuse])

        fsr3, fsr2, fsr1 = fsr[0], fsr[1], fsr[2]

        # de4 = self.de4(fsr4)
        de3 = self.de3(fsr3)
        de2 = self.de2(fsr2)
        de1 = self.de1(fsr1)

        side_output3 = self.compress3(de3)
        side_output2 = self.compress2(de2)
        side_output1 = self.compress1(de1)
        # side_output1 = self.compress1(de1)

        side_output3 = F.interpolate(side_output3, scale_factor=8, mode="bilinear", align_corners=True)
        side_output2 = F.interpolate(side_output2, scale_factor=4, mode="bilinear", align_corners=True)
        side_output1 = F.interpolate(side_output1, scale_factor=2, mode="bilinear", align_corners=True)

        side_cat = torch.cat([side_output3, side_output2, side_output1], dim=1)

        output = self.output_layer(side_cat)

        return torch.sigmoid(output)


if __name__ == "__main__":
    x = torch.randn([1, 3, 480, 320])
    model = MyModel()
    o = model(x)
    print(o.shape)

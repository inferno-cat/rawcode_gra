import torch
import torch.nn as nn
from torchvision import models


class VGG(nn.Module):
    def __init__(self, name="vgg16_bn", pretrained=False):
        super(VGG, self).__init__()

        weights_dict = {
            "vgg16": models.VGG16_Weights.DEFAULT,
            "vgg16_bn": models.VGG16_BN_Weights.DEFAULT,
            "vgg19": models.VGG19_Weights.DEFAULT,
            "vgg19_bn": models.VGG19_BN_Weights.DEFAULT,
        }

        if name == "vgg16":
            if pretrained:
                weights = weights_dict[name]
            else:
                weights = None
            pretrained_model = models.vgg16(weights=weights)
        elif name == "vgg16_bn":
            if pretrained:
                weights = weights_dict[name]
            else:
                weights = None
            pretrained_model = models.vgg16_bn(weights=weights)
        elif name == "vgg19":
            if pretrained:
                weights = weights_dict[name]
            else:
                weights = None
            pretrained_model = models.vgg19(weights=weights)
        elif name == "vgg19_bn":
            if pretrained:
                weights = weights_dict[name]
            else:
                weights = None
            pretrained_model = models.vgg19_bn(weights=weights)
        else:
            raise NameError

        self.conv1 = self.extract_layer(pretrained_model, name, 1)
        self.conv2 = self.extract_layer(pretrained_model, name, 2)
        self.conv3 = self.extract_layer(pretrained_model, name, 3)
        self.conv4 = self.extract_layer(pretrained_model, name, 4)
        self.conv5 = self.extract_layer(pretrained_model, name, 5)

    def forward(self, x):
        feature1 = self.conv1(x)
        feature2 = self.conv2(feature1)
        feature3 = self.conv3(feature2)
        feature4 = self.conv4(feature3)
        feature5 = self.conv5(feature4)

        return [feature1, feature2, feature3, feature4, feature5]

    def extract_layer(self, model, backbone_mode, ind):
        if backbone_mode == "vgg16":
            index_dict = {1: (0, 4), 2: (4, 9), 3: (9, 16), 4: (16, 23), 5: (23, 30)}
        elif backbone_mode == "vgg16_bn":
            index_dict = {1: (0, 6), 2: (6, 13), 3: (13, 23), 4: (23, 33), 5: (33, 43)}
        elif backbone_mode == "vgg19":
            index_dict = {1: (0, 4), 2: (4, 9), 3: (9, 18), 4: (18, 27), 5: (27, 36)}
        elif backbone_mode == "vgg19_bn":
            index_dict = {1: (0, 6), 2: (6, 13), 3: (13, 26), 4: (26, 39), 5: (39, 52)}
        else:
            raise NameError

        start, end = index_dict[ind]
        modified_model = nn.Sequential(*list(model.features.children())[start:end])
        return modified_model


if __name__ == "__main__":
    model = VGG(pretrained=False)
    print(model)

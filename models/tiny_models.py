import os

import torch
import torch.nn as nn
from torchvision import models


# def MobileNetSmallV3(pretrained=False):
#     base_model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
#     for param in base_model.parameters():
#         param.requires_grad = False
#
#     base_model.classifier = nn.Sequential(
#         nn.Linear(576, 1024),
#         nn.Hardswish(),
#         nn.Dropout(p=0.2, inplace=True),
#         nn.Linear(1024, 10)
#     )
#
#     base_model.classifier[-1].is_classifier = True
#
#     if pretrained:
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         dir_path = os.path.dirname(os.path.realpath(__file__))
#         base_model.load_state_dict(torch.load(dir_path + "/../checkpoints/mobilenet_v3_small_cifar10.pt",
#                                               map_location=device))
#
#     return base_model

class MobileNetSmallV3(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        base_model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)

        self.features = base_model.features
        self.avgpool = base_model.avgpool
        self.classifier = nn.Sequential(
            nn.Linear(576, 1024),
            nn.Hardswish(),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1024, 10)
        )

        for param in self.features.parameters():
            param.requires_grad = False

        self.classifier[-1].is_classifier = True

        if pretrained:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            dir_path = os.path.dirname(os.path.realpath(__file__))
            self.backbone.load_state_dict(torch.load(dir_path + "/../checkpoints/mobilenet_v3_small_cifar10.pt",
                                                     map_location=device))

    def forward(self, x):
        with torch.no_grad():
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)

        return self.classifier(x)

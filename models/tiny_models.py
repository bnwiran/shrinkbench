import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class MobileNetSmallV3(nn.Module):
    def __init__(self, pretrained: Optional[bool] = True):
        super().__init__()
        feature_extractor = models.mobilenet_v3_small(weigths=models.MobileNet_V3_Small_Weights.DEFAULT)

        for param in feature_extractor.parameters():
            param.requires_grad = False

        feature_extractor.classifier = nn.Sequential(
            nn.Linear(576, 1024),
            nn.Hardswish(),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1024, 10)
        )

        feature_extractor.classifier[3].is_classifier = True
        self.feature_extractor = feature_extractor

        if pretrained:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            dir_path = os.path.dirname(os.path.realpath(__file__))
            self.feature_extractor.load_state_dict(torch.load(dir_path + "/../checkpoints/mobilenet_v3_small_cifar10.pt",
                                                              map_location=device))

    def forward(self, inputs):
        return self.feature_extractor(inputs)

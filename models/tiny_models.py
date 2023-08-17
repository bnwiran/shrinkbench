import os

import torch
import torch.nn as nn
from torchvision import models


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

        if pretrained:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            dir_path = os.path.dirname(os.path.realpath(__file__))
            state_dict = torch.load(dir_path + "/../checkpoints/mobilenet_v3_small_cifar10.pt",
                                    map_location=device)
            mode_state_dict = state_dict['model_state_dict']
            self.load_state_dict(mode_state_dict)

        self.classifier[-1].is_classifier = True

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

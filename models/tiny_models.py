import os

import lightning.pytorch as pl
import torch
import torch.nn as nn
from torch import optim, Tensor
from torch.nn import functional as F
from torchmetrics.classification import MulticlassAccuracy
from torchvision import models


class MobileNetSmallV3Light(pl.LightningModule):
    def __init__(self, num_classes: int, pretrained=False):
        super().__init__()
        assert num_classes > 1, "Number of classes must be greater than" + str(num_classes)
        base_model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)

        self.features = base_model.features
        self.avgpool = base_model.avgpool
        self.classifier = nn.Sequential(
            nn.Linear(576, 1024),
            nn.Hardswish(),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1024, num_classes)
        )

        if pretrained:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            dir_path = os.path.dirname(os.path.realpath(__file__))
            state_dict = torch.load(dir_path + "/../checkpoints/mobilenet_v3_small_cifar10.pt",
                                    map_location=device)
            mode_state_dict = state_dict['model_state_dict']
            self.load_state_dict(mode_state_dict)

        self.classifier[-1].is_classifier = True

        self._create_metrics(num_classes)

    def _create_metrics(self, num_classes):
        self.train_acc1 = MulticlassAccuracy(num_classes=num_classes)
        self.train_acc5 = MulticlassAccuracy(num_classes=num_classes, top_k=5)
        self.val_acc1 = MulticlassAccuracy(num_classes=num_classes)
        self.val_acc5 = MulticlassAccuracy(num_classes=num_classes, top_k=5)
        self.test_acc1 = MulticlassAccuracy(num_classes=num_classes)
        self.test_acc5 = MulticlassAccuracy(num_classes=num_classes, top_k=5)

    def training_step(self, batch, batch_idx) -> Tensor:
        x, y = batch
        logits = self(x)

        loss = nn.functional.cross_entropy(logits, y)

        self.train_acc1(logits, y)
        self.train_acc5(logits, y)
        self.log('train_acc1', self.train_acc1, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_acc5', self.train_acc5, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        loss = F.cross_entropy(logits, y)

        self.val_acc1(logits, y)
        self.val_acc5(logits, y)
        self.log('val_acc1', self.val_acc1, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_acc5', self.val_acc5, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_loss", loss, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        loss = F.cross_entropy(logits, y)

        self.test_acc1(logits, y)
        self.test_acc5(logits, y)
        self.log('test_acc1', self.test_acc1, prog_bar=True, on_step=False, on_epoch=True)
        self.log('test_acc5', self.test_acc5, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test_loss", loss, prog_bar=True)

        return loss

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)

        return logits

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


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

import os

import lightning.pytorch as pl
import torch
import torch.nn as nn
from torch import optim, Tensor
from torch.nn import functional as F
from torchmetrics.classification import MulticlassAccuracy
from torchvision import models


class MobileNetSmallV3(pl.LightningModule):
    def __init__(self, num_classes: int, pretrained: str = None) -> None:
        super().__init__()
        assert num_classes > 1, "Number of classes must be greater than" + str(num_classes)

        if pretrained is not None and pretrained.lower() == 'imagenet':
            base_model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        else:
            base_model = models.mobilenet_v3_small()

        self.features = base_model.features
        self.avgpool = base_model.avgpool
        if num_classes == 1000:
            self.classifier = base_model.classifier
        else:
            self.classifier = nn.Sequential(
                nn.Linear(576, 1024),
                nn.Hardswish(),
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(1024, num_classes)
            )

        self.classifier[-1].is_classifier = True

        if pretrained is not None and pretrained.lower() != 'imagenet':
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            self.load_state_dict(torch.load(pretrained, map_location=device)['model_state_dict'])

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
        optimizer = optim.Adam(self.parameters(), lr=5e-4)
        return optimizer


class ShuffleNetV2(pl.LightningModule):
    def __init__(self, num_classes: int, pretrained: str = None):
        super().__init__()
        assert num_classes > 1, "Number of classes must be greater than" + str(num_classes)

        if pretrained is not None and pretrained.lower() == 'imagenet':
            model = models.shufflenet_v2_x0_5(weights=models.ShuffleNet_V2_X0_5_Weights.DEFAULT)
        else:
            model = models.shufflenet_v2_x0_5()

        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        model.fc.is_classifier = True

        if pretrained is not None and pretrained.lower() != 'imagenet':
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            model.load_state_dict(torch.load(pretrained, map_location=device))

        self.model = model
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
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        return optimizer


class SqueezeNetV1(pl.LightningModule):
    def __init__(self, num_classes: int, pretrained: str = None):
        super().__init__()
        assert num_classes > 1, "Number of classes must be greater than" + str(num_classes)

        if pretrained is not None and pretrained.lower() == 'imagenet':
            base_model = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.DEFAULT)
        else:
            base_model = models.squeezenet1_1()

        self.features = base_model.features
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=False),
            nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )
        self.classifier[1].is_classifier = True

        if pretrained is not None and pretrained.lower() != 'imagenet':
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            self.load_state_dict(torch.load(pretrained, map_location=device))

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
        x = self.classifier(x)
        logits = torch.flatten(x, 1)

        return logits

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

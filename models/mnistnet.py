"""Small CNN designed for MNIST, intended for debugging purposes

[description]
"""

import lightning.pytorch as pl
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, optim
from torchmetrics.classification import MulticlassAccuracy


class MnistNet(pl.LightningModule):
    """Small network designed for Mnist debugging
    """

    def __init__(self, pretrained=False) -> None:
        assert not pretrained, f"{self.__class__.__name__} does not support pretrained weights"
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)
        self.fc2.is_classifier = True

        num_classes = 10
        self.train_acc1 = MulticlassAccuracy(num_classes=num_classes)
        self.train_acc5 = MulticlassAccuracy(num_classes=num_classes, top_k=5)
        self.val_acc1 = MulticlassAccuracy(num_classes=num_classes)
        self.val_acc5 = MulticlassAccuracy(num_classes=num_classes, top_k=5)
        self.test_acc1 = MulticlassAccuracy(num_classes=num_classes)
        self.test_acc5 = MulticlassAccuracy(num_classes=num_classes, top_k=5)

    def training_step(self, batch, batch_idx) -> Tensor:
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

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
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)

        return logits

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class LeNet(nn.Module):
    def __init__(self, pretrained=False):
        assert not pretrained, f"{self.__class__.__name__} does not support pretrained weights"
        super(LeNet, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square conv kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5x5 image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

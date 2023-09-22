import lightning.pytorch as pl
import torch
import torch.nn as nn
from torch import optim, Tensor
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, SequentialLR, ConstantLR
from torchmetrics.classification import MulticlassAccuracy
from torchvision import models


from . import utils


class MobileNetSmallV3(pl.LightningModule):
    def __init__(self, num_classes: int, pretrained: str = None) -> None:
        super().__init__()
        assert num_classes > 1, "Number of classes must be greater than" + str(num_classes)

        if pretrained is not None and pretrained.lower() == 'imagenet':
            base_model = models.get_model('mobilenet_v3_small', weight=models.MobileNet_V3_Small_Weights.DEFAULT,
                                          num_classes=num_classes)
        else:
            base_model = models.get_model('mobilenet_v3_small', num_classes=num_classes)

        base_model.classifier[-1].is_classifier = True
        self.base_model = base_model

        if pretrained is not None and pretrained.lower() != 'imagenet':
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            self.load_state_dict(torch.load(pretrained, map_location=device)['model_state_dict'])

        self._create_metrics(num_classes)
        self.learning_rate = 0.064
        self.weight_decay = 1e-5
        self.momentum = 0.9
        self.lr_step_size = 2
        self.lr_gamma = 0.973

    def _create_metrics(self, num_classes):
        self.train_acc1 = MulticlassAccuracy(num_classes=num_classes, average='micro')
        self.train_acc5 = MulticlassAccuracy(num_classes=num_classes, average='micro', top_k=5)
        self.val_acc1 = MulticlassAccuracy(num_classes=num_classes, average='micro')
        self.val_acc5 = MulticlassAccuracy(num_classes=num_classes, average='micro', top_k=5)
        self.test_acc1 = MulticlassAccuracy(num_classes=num_classes, average='micro')
        self.test_acc5 = MulticlassAccuracy(num_classes=num_classes, average='micro', top_k=5)

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
        return self.base_model(x)

    def configure_optimizers(self):
        parameters = utils.set_weight_decay(
            self.base_model,
            self.weight_decay
        )
        optimizer = optim.RMSprop(parameters, lr=self.learning_rate, momentum=self.momentum,
                                  weight_decay=self.weight_decay, eps=0.0316, alpha=0.9)
        scheduler = StepLR(optimizer, step_size=self.lr_step_size, gamma=self.lr_gamma)
        return [optimizer], [scheduler]


class ShuffleNetV2(pl.LightningModule):
    def __init__(self, num_classes: int, pretrained: str = None):
        super().__init__()
        assert num_classes > 1, "Number of classes must be greater than" + str(num_classes)

        if pretrained is not None and pretrained.lower() == 'imagenet':
            base_model = models.get_model('shufflenet_v2_x0_5', weight=models.ShuffleNet_V2_X0_5_Weights.DEFAULT,
                                          num_classes=num_classes)
        else:
            base_model = models.get_model('shufflenet_v2_x0_5', num_classes=num_classes)

        base_model.fc.is_classifier = True

        if pretrained is not None and pretrained.lower() != 'imagenet':
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            base_model.load_state_dict(torch.load(pretrained, map_location=device))

        self.label_smoothing = 0.1
        self.base_model = base_model
        self._create_metrics(num_classes)

    def _create_metrics(self, num_classes):
        self.train_acc1 = MulticlassAccuracy(num_classes=num_classes, average='micro')
        self.train_acc5 = MulticlassAccuracy(num_classes=num_classes, average='micro', top_k=5)
        self.val_acc1 = MulticlassAccuracy(num_classes=num_classes, average='micro')
        self.val_acc5 = MulticlassAccuracy(num_classes=num_classes, average='micro', top_k=5)
        self.test_acc1 = MulticlassAccuracy(num_classes=num_classes, average='micro')
        self.test_acc5 = MulticlassAccuracy(num_classes=num_classes, average='micro', top_k=5)

    def training_step(self, batch, batch_idx) -> Tensor:
        x, y = batch
        logits = self(x)

        loss = F.cross_entropy(logits, y, label_smoothing=self.label_smoothing)

        if y.ndim == 2:
            y = y.max(dim=1)[1]

        self.train_acc1(logits, y)
        self.train_acc5(logits, y)
        self.log('train_acc1', self.train_acc1, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_acc5', self.train_acc5, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        loss = F.cross_entropy(logits, y, label_smoothing=self.label_smoothing)

        self.val_acc1(logits, y)
        self.val_acc5(logits, y)
        self.log('val_acc1', self.val_acc1, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_acc5', self.val_acc5, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_loss", loss, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        loss = F.cross_entropy(logits, y, label_smoothing=self.label_smoothing)

        self.test_acc1(logits, y)
        self.test_acc5(logits, y)
        self.log('test_acc1', self.test_acc1, prog_bar=True, on_step=False, on_epoch=True)
        self.log('test_acc5', self.test_acc5, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test_loss", loss, prog_bar=True)

        return loss

    def forward(self, x):
        return self.base_model(x)

    def configure_optimizers(self):
        weight_decay = 2e-5
        lr_warmup_decay = 0.01
        lr_warmup_epochs = 5
        epochs = 700
        lr_min = 0
        norm_weight_decay = 0
        parameters = utils.set_weight_decay(
            self.base_model,
            weight_decay,
            norm_weight_decay=norm_weight_decay
        )
        optimizer = optim.SGD(parameters, lr=0.5, momentum=0.9, weight_decay=weight_decay, nesterov=False)
        main_lr_scheduler = CosineAnnealingLR(optimizer, T_max=epochs - lr_warmup_epochs, eta_min=lr_min)
        warmup_lr_scheduler = ConstantLR(optimizer, factor=lr_warmup_decay, total_iters=lr_warmup_epochs)
        lr_scheduler = SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[lr_warmup_epochs]
        )
        return [optimizer], [lr_scheduler]


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

        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

        if pretrained is not None and pretrained.lower() != 'imagenet':
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            self.load_state_dict(torch.load(pretrained, map_location=device))

        self._create_metrics(num_classes)

    def _create_metrics(self, num_classes):
        self.train_acc1 = MulticlassAccuracy(num_classes=num_classes, average='micro')
        self.train_acc5 = MulticlassAccuracy(num_classes=num_classes, average='micro', top_k=5)
        self.val_acc1 = MulticlassAccuracy(num_classes=num_classes, average='micro')
        self.val_acc5 = MulticlassAccuracy(num_classes=num_classes, average='micro', top_k=5)
        self.test_acc1 = MulticlassAccuracy(num_classes=num_classes, average='micro')
        self.test_acc5 = MulticlassAccuracy(num_classes=num_classes, average='micro', top_k=5)

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

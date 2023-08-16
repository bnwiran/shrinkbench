import pathlib
import sys
import time
from abc import abstractmethod
from typing import Optional

import torch
import torchvision.models
from torch import nn
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.backends import cudnn
from tqdm import tqdm
import json

from .base import Experiment
from .. import datasets
from .. import models
from ..metrics import correct
from ..models.head import mark_classifier
from ..util import printc, OnlineStats


class TrainingExperiment(Experiment):
    default_dl_kwargs = {'batch_size': 128,
                         'pin_memory': False,
                         'num_workers': 8
                         }

    default_train_kwargs = {'optim': 'SGD',
                            'epochs': 30,
                            'lr': 1e-3,
                            }

    def __init__(self,
                 dataset,
                 model,
                 seed=42,
                 path=None,
                 dl_kwargs: dict = None,
                 train_kwargs: dict = None,
                 debug=False,
                 pretrained=False,
                 resume=None,
                 resume_optim=False,
                 save_freq=10):

        # Default children kwargs
        super().__init__(seed)

        self.trainer = None
        self.epochs = None
        self.resume = None
        self.model = None
        self.val_dl = None
        self.train_dl = None
        self.val_dataset = None
        self.train_dataset = None
        self.optim = None

        if train_kwargs is None:
            train_kwargs = dict()
        if dl_kwargs is None:
            dl_kwargs = dict()

        dl_kwargs = {**self.default_dl_kwargs, **dl_kwargs}
        train_kwargs = {**self.default_train_kwargs, **train_kwargs}

        params = locals()
        params['dl_kwargs'] = dl_kwargs
        params['train_kwargs'] = train_kwargs
        self.add_params(**params)

        self._build_dataloader(dataset, **dl_kwargs)
        self._build_model(model, pretrained, resume)
        self._build_train(resume_optim=resume_optim, **train_kwargs)

        self.path = path
        self.save_freq = save_freq

    def setup(self):
        self.freeze()
        printc(f"Running {repr(self)}", color='YELLOW')
        self._build_logging(self.train_metrics, self.path)
        self.trainer = Trainer(self.model, self.path, self.optim, self.loss_func, self.save_freq, self.log)

    @abstractmethod
    def run(self):
        pass

    def wrapup(self):
        pass

    def _build_dataloader(self, dataset, **dl_kwargs):
        constructor = getattr(datasets, dataset)
        self.train_dataset = constructor(train=True)
        self.val_dataset = constructor(train=False)
        self.train_dl = DataLoader(self.train_dataset, shuffle=True, **dl_kwargs)
        self.val_dl = DataLoader(self.val_dataset, shuffle=False, **dl_kwargs)

    def _build_model(self, model, pretrained=True, resume=None):
        if isinstance(model, str):
            if hasattr(models, model):
                model = getattr(models, model)(pretrained=pretrained)

            elif hasattr(torchvision.models, model):
                # https://pytorch.org/docs/stable/torchvision/models.html
                model = getattr(torchvision.models, model)(pretrained=pretrained)
                mark_classifier(model)  # add is_classifier attribute
            else:
                raise ValueError(f"Model {model} not available in custom models or torchvision models")

        self.model = model

        if resume is not None:
            self.resume = pathlib.Path(self.resume)
            assert self.resume.exists(), "Resume path does not exist"
            previous = torch.load(self.resume)
            self.model.load_state_dict(previous['model_state_dict'])

    def _build_train(self, optim, epochs, resume_optim=False, **optim_kwargs):
        default_optim_kwargs = {
            'SGD': {'momentum': 0.9, 'nesterov': True, 'lr': 1e-3},
            'Adam': {'momentum': 0.9, 'betas': (.9, .99), 'lr': 1e-4}
        }

        self.epochs = epochs

        # Optim
        if isinstance(optim, str):
            constructor = getattr(torch.optim, optim)
            if optim in default_optim_kwargs:
                optim_kwargs = {**default_optim_kwargs[optim], **optim_kwargs}
            optim = constructor(self.model.parameters(), **optim_kwargs)

        self.optim = optim

        if resume_optim:
            assert hasattr(self, "resume"), "Resume must be given for resume_optim"
            previous = torch.load(self.resume)
            self.optim.load_state_dict(previous['optim_state_dict'])

        # Assume classification experiment
        self.loss_func = nn.CrossEntropyLoss()

    def run_epochs(self):
        since = time.time()

        time_logger = lambda _: self.log(timestamp=time.time() - since)
        epoch_logger = lambda epoch: self.log_epoch(epoch)

        self.trainer.fit(self.train_dl, self.val_dl, self.epochs, [time_logger, epoch_logger])

    @property
    def train_metrics(self):
        return ['epoch', 'timestamp',
                'train_loss', 'train_acc1', 'train_acc5',
                'val_loss', 'val_acc1', 'val_acc5',
                ]

    def __repr__(self):
        if not isinstance(self.params['model'], str) and isinstance(self.params['model'], torch.nn.Module):
            self.params['model'] = self.params['model'].__module__

        assert isinstance(self.params['model'], str), f"\nUnexpected model inputs: {self.params['model']}"
        return json.dumps(self.params, indent=4)


class Trainer:
    def __init__(self, model: Module, path: pathlib.Path, optim, loss_func, save_freq: int = 10,
                 log: callable = None) -> None:
        self.model = model
        self.path = path
        self.save_freq = save_freq
        self.optim = optim
        self.log = log
        self.loss_func = loss_func
        self._set_device()

    def _set_device(self):
        # Torch CUDA config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not torch.cuda.is_available():
            printc("GPU NOT AVAILABLE, USING CPU!", color="ORANGE")
        self.model.to(self.device)
        cudnn.benchmark = True  # For fast training.

    def fit(self, train_dl, valid_dl, epochs=25, callbacks: Optional[list[callable]] = None):
        epoch = 0
        best_acc = 0

        try:
            for epoch in range(1, epochs + 1):
                train_metrics = self._train_epoch(train_dl, epoch, epochs)
                val_metrics = self.evaluate(valid_dl)
                val_acc1 = val_metrics['val_acc1']

                if val_acc1 > best_acc:
                    best_acc = val_acc1
                    self._checkpoint(epoch)

                if epoch % self.save_freq == 0:
                    self._checkpoint(epoch)

                # TODO Early stopping
                # TODO ReduceLR on plateau?

                if self.log is not None:
                    self.log(**train_metrics)
                    self.log(**val_metrics)

                if callbacks is not None:
                    for callback in callbacks:
                        callback(epoch)

        except KeyboardInterrupt:
            printc(f"\nInterrupted at epoch {epoch}. Tearing Down", color='RED')

    def _train_epoch(self, train_dl, epoch, epochs):
        total_loss = OnlineStats()
        acc1 = OnlineStats()
        acc5 = OnlineStats()

        epoch_iter = tqdm(train_dl, file=sys.stdout, ascii=' >=')
        epoch_iter.set_description(f"TRAIN Epoch {epoch}/{epochs}")

        self.model.train()
        with torch.set_grad_enabled(True):
            for i, (x, y) in enumerate(epoch_iter, start=1):
                x, y = x.to(self.device), y.to(self.device)
                yhat = self.model(x)
                loss = self.loss_func(yhat, y)
                loss.backward()

                self.optim.step()
                self.optim.zero_grad()

                c1, c5 = correct(yhat, y, (1, 5))
                total_loss.add(loss.item() / train_dl.batch_size)
                acc1.add(c1 / train_dl.batch_size)
                acc5.add(c5 / train_dl.batch_size)

                epoch_iter.set_postfix(loss=total_loss.mean, top1=acc1.mean, top5=acc5.mean)

        return {
            'train_loss': total_loss.mean,
            'train_acc1': acc1.mean,
            'train_acc5': acc5.mean,
        }

    def evaluate(self, valid_dl):
        total_loss = OnlineStats()
        acc1 = OnlineStats()
        acc5 = OnlineStats()

        epoch_iter = tqdm(valid_dl, file=sys.stdout, ascii=' >=')
        epoch_iter.set_description("VAL")

        self.model.eval()
        with torch.no_grad():
            for i, (x, y) in enumerate(epoch_iter, start=1):
                x, y = x.to(self.device), y.to(self.device)
                yhat = self.model(x)
                loss = self.loss_func(yhat, y)

                c1, c5 = correct(yhat, y, (1, 5))
                total_loss.add(loss.item() / valid_dl.batch_size)
                acc1.add(c1 / valid_dl.batch_size)
                acc5.add(c5 / valid_dl.batch_size)

                epoch_iter.set_postfix(loss=total_loss.mean, top1=acc1.mean, top5=acc5.mean)

        return {
            'val_loss': total_loss.mean,
            'val_acc1': acc1.mean,
            'val_acc5': acc5.mean,
        }

    def _checkpoint(self, epoch):
        checkpoint_path = self.path / 'checkpoints'
        checkpoint_path.mkdir(exist_ok=True, parents=True)
        model_name = checkpoint_path / f'checkpoint-{epoch}.pt'
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optim_state_dict': self.optim.state_dict()
        }, model_name)

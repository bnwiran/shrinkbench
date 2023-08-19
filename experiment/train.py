import json
import pathlib
from abc import abstractmethod

import lightning.pytorch as pl
import torch
import torchvision.models
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from .base import Experiment
from .. import datasets
from .. import models
from ..models.head import mark_classifier
from ..util import printc


class TrainingExperiment(Experiment):
    default_dl_kwargs = {'batch_size': 128,
                         'pin_memory': False,
                         'num_workers': 8
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

        if dl_kwargs is None:
            dl_kwargs = dict()

        dl_kwargs = {**self.default_dl_kwargs, **dl_kwargs}

        params = locals()
        params['dl_kwargs'] = dl_kwargs
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

        checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_acc1")
        self.trainer = pl.Trainer(default_root_dir=self.path, max_epochs=self.epochs, callbacks=[checkpoint_callback])

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

    def _build_train(self, epochs, resume_optim=False):
        self.epochs = epochs

    def run_epochs(self):
        self.trainer.fit(model=self.model, train_dataloaders=self.train_dl, val_dataloaders=self.val_dl)

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

import json
import logging
import pathlib

import lightning.pytorch as pl
import numpy as np
import torch
import torchvision.models
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from .base import Experiment
from .. import datasets
from .. import models
from ..models.head import mark_classifier


class TrainingExperiment(Experiment):
    default_dl_kwargs = {
        'batch_size': 128,
        'pin_memory': False,
        'num_workers': 8
    }
    default_model_kwargs = {
        'pretrained': False
    }

    def __init__(self,
                 name: str,
                 dataset,
                 model,
                 resume_exp: bool = False,
                 seed=42,
                 path=None,
                 model_kwargs: dict = None,
                 dl_kwargs: dict = None,
                 train_kwargs: dict = None,
                 debug=False,
                 resume=None,
                 resume_optim=False,
                 save_freq=10):

        # Default children kwargs
        super().__init__(name, path, resume_exp, seed)

        self.checkpoint_callback = None
        self.trainer = None
        self.epochs = None
        self.resume = None
        self.model = None
        self.val_dl = None
        self.train_dl = None
        self.optim = None

        if model_kwargs is None:
            model_kwargs = dict()

        if dl_kwargs is None:
            dl_kwargs = dict()

        dl_kwargs = {**self.default_dl_kwargs, **dl_kwargs}
        model_kwargs = {**self.default_model_kwargs, **model_kwargs}

        params = locals()
        params['dl_kwargs'] = dl_kwargs
        params['model_kwargs'] = model_kwargs
        self.add_params(**params)
        self._build_dataloaders(dataset, **dl_kwargs)
        self._build_model(model, resume, **model_kwargs)
        self._build_train(resume_optim=resume_optim, **train_kwargs)
        self.save_freq = save_freq

    def setup(self):
        logging.info("Setting up the experiment")
        self.freeze()
        self.save_params()

    def run(self):
        self.checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_acc1")
        early_stop_callback = EarlyStopping(monitor="val_acc1", min_delta=0.00, patience=3, verbose=False,
                                            mode="max")
        self.trainer = pl.Trainer(default_root_dir=self.path, max_epochs=self.epochs,
                                  callbacks=[self.checkpoint_callback, early_stop_callback])

    def wrapup(self):
        pass

    def _build_dataloaders(self, dataset, **dl_kwargs):
        constructor = getattr(datasets, dataset)
        dataset = constructor(train=True)
        targets = dataset.targets
        train_idx, val_idx = train_test_split(np.arange(len(targets)), test_size=dataset.val_size, random_state=42,
                                              shuffle=True, stratify=targets)
        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)
        self.train_dl = DataLoader(train_dataset, shuffle=True, **dl_kwargs)
        self.val_dl = DataLoader(val_dataset, shuffle=False, **dl_kwargs)

    def _build_model(self, model, resume=None, **model_kwargs):
        if isinstance(model, str):
            if hasattr(models, model):
                model = getattr(models, model)(**model_kwargs)

            elif hasattr(torchvision.models, model):
                # https://pytorch.org/docs/stable/torchvision/models.html
                model = getattr(torchvision.models, model)(**model_kwargs)
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
        best_model_path = self.checkpoint_callback.best_model_path
        ckpt_path = None if best_model_path == '' else best_model_path
        self.trainer.fit(model=self.model, train_dataloaders=self.train_dl, val_dataloaders=self.val_dl,
                         ckpt_path=ckpt_path)

    def __repr__(self):
        if not isinstance(self.params['model'], str) and isinstance(self.params['model'], torch.nn.Module):
            self.params['model'] = self.params['model'].__module__

        assert isinstance(self.params['model'], str), f"\nUnexpected model inputs: {self.params['model']}"
        return json.dumps(self.params, indent=4)

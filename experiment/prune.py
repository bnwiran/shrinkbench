import json
import logging
import pathlib
from typing import Union

import lightning.pytorch as pl
import torch
import torchvision
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from torch import nn
from torch.utils.data import DataLoader, default_collate

from ..datasets.transforms import get_mixup_cutmix
from . import Experiment
from .. import datasets
from .. import models
from .. import strategies
from ..metrics import model_size, flops
from ..models.head import mark_classifier


class PruningExperiment(Experiment):
    default_dl_kwargs = {
        'batch_size': 128,
        'pin_memory': True,
        'num_workers': 8,
        'mixup_alpha': 0.0,
        'cutmix_alpha': 0.0,
    }
    default_ds_kwargs = {
        'val_resize_size': 256,
        'val_crop_size': 224,
        'train_crop_size': 224,
        'interpolation': 'bilinear',
        'backend': 'PIL',
        'use_v2': False,
        'weights': None,
        'auto_augment': None,
        'random_erase': 0.0,
        'ra_magnitude': 9,
        'augmix_severity': 3,
        'cache_dataset': False,
        'test_only': False,
        'ra_sampler': False,
        'distributed': False,
        'ra_reps': 3,
    }
    default_model_kwargs = {
        'pretrained': None
    }
    default_opt_kwargs = {
        'learning_rate': 0.1,
        'weight_decay': 1e-4,
        'norm_weight_decay': None,
        'bias_weight_decay': None,
        'momentum': 0.9,
        'lr_step_size': 30,
        'lr_gamma': 0.1,
        'lr_min': 0.0,
        'lr_scheduler': 'steplr',
        'lr_warmup_method': 'constant',
        'lr-warmup_decay': '0.01',
        'lr_warmup_epochs': 0.0,
        'label_smoothing': 0.0,
    }

    def __init__(self,
                 name: str,
                 dataset: str,
                 model: str,
                 strategy: str,
                 compression: Union[list[int], int],
                 resume_exp: bool = False,
                 seed: int = 42,
                 path: str = None,
                 model_kwargs=None,
                 ds_kwargs=None,
                 dl_kwargs=None,
                 train_kwargs=None,
                 debug: bool = False,
                 resume: bool = None,
                 resume_optim: bool = False,
                 save_freq: int = 10) -> None:

        super().__init__(name, path, resume_exp, seed)

        self.resume = None
        self.trainer = None

        if model_kwargs is None:
            model_kwargs = dict()

        if ds_kwargs is None:
            ds_kwargs = dict()

        if dl_kwargs is None:
            dl_kwargs = dict()

        ds_kwargs = {**self.default_ds_kwargs, **ds_kwargs}
        dl_kwargs = {**self.default_dl_kwargs, **dl_kwargs}
        model_kwargs = {**self.default_model_kwargs, **model_kwargs}

        params = locals()
        params['ds_kwargs'] = ds_kwargs
        params['dl_kwargs'] = dl_kwargs
        params['model_kwargs'] = model_kwargs

        self.add_params(**params)
        self._build_dataloaders(dataset, ds_kwargs, **dl_kwargs)
        self._build_model(model, resume, **model_kwargs)
        self._build_train(resume_optim=resume_optim, **train_kwargs)

        self.pruning = None
        self.strategy = strategy
        self.compression = [compression] if isinstance(compression, int) else compression
        self.add_params(strategy=strategy, compression=compression)
        self.save_freq = save_freq

    def setup(self):
        logging.info("Setting up the experiment")
        self.freeze()
        self.save_params()

    def run(self):
        constructor = getattr(strategies, self.strategy)
        checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_acc1", mode='max', save_last=True,
                                              dirpath=self.path / 'checkpoints')
        early_stop_callback = EarlyStopping(monitor="val_acc1", min_delta=0.00, patience=5, verbose=False,
                                            mode="max")
        trainer = pl.Trainer(default_root_dir=self.path, max_epochs=self.epochs,
                             callbacks=[checkpoint_callback])  # callbacks=[checkpoint_callback, early_stop_callback])
        self.trainer = trainer

        for c in self.compression:
            logging.info(f'Running pruning experiment with compression {c}')
            x, y = next(iter(self.train_dl))

            self.pruning = constructor(self.model, x, y, compression=c)
            self.pruning.apply()
            self._fit()
            self._save_metrics(c)

            # Until finding another way, this hacking is necessary
            trainer.save_checkpoint(self.path / 'checkpoints' / f'compression={c}.ckpt')
            early_stop_callback.wait_count = 0
            trainer.should_stop = False

    def wrapup(self):
        pass

    def _fit(self):
        self.trainer.fit(model=self.model, train_dataloaders=self.train_dl, val_dataloaders=self.val_dl,
                         ckpt_path='last')

    def _build_dataloaders(self, dataset, ds_kwargs, **dl_kwargs):
        constructor = getattr(datasets, dataset)
        train_dataset, val_dataset, train_sampler, test_sampler = constructor(args=ds_kwargs)
        num_classes = len(train_dataset.classes)
        collate_fn = self._get_collate_fn(dl_kwargs, num_classes)

        batch_size = dl_kwargs['batch_size']
        num_workers = dl_kwargs['num_workers']
        pin_memory = dl_kwargs['pin_memory']
        self.train_dl = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers,
                                   pin_memory=pin_memory, collate_fn=collate_fn)
        self.val_dl = DataLoader(val_dataset, batch_size=batch_size, sampler=test_sampler, num_workers=num_workers,
                                 pin_memory=pin_memory)

    def _get_collate_fn(self, dl_kwargs, num_classes):
        mixup_cutmix = get_mixup_cutmix(
            mixup_alpha=dl_kwargs['mixup_alpha'], cutmix_alpha=dl_kwargs['cutmix_alpha'], num_categories=num_classes,
            use_v2=dl_kwargs['use_v2']
        )
        if mixup_cutmix is not None:
            def collate_fn(batch):
                return mixup_cutmix(*default_collate(batch))
        else:
            collate_fn = default_collate
        return collate_fn

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

    def _save_metrics(self, compression):
        metrics = self._model_metrics()
        self._log_csv(metrics)

        if compression > 0:
            with open(self.path / f'metrics-{compression}.json', 'w') as f:
                json.dump(metrics, f, indent=4)

            summary = self.pruning.summary()
            summary_path = self.path / f'masks_summary_{compression}.csv'
            summary.to_csv(summary_path)

    def _model_metrics(self):
        metrics = {}
        size, size_nz = model_size(self.model)
        metrics['size'] = size
        metrics['size_nz'] = size_nz
        metrics['compression_ratio'] = size / size_nz

        x, _ = next(iter(self.val_dl))
        x = x.to(self.model.device)

        ops, ops_nz = flops(self.model, x)
        metrics['flops'] = ops
        metrics['flops_nz'] = ops_nz
        metrics['theoretical_speedup'] = ops / ops_nz

        val_metrics = self.trainer.validate(self.model, self.val_dl)[0]
        metrics['loss'] = val_metrics['val_loss']
        metrics['val_acc1'] = val_metrics['val_acc1']
        metrics['val_acc5'] = val_metrics['val_acc5']

        return metrics

    def _log_csv(self, metrics: dict) -> None:
        summary_path = self.path / f'metrics_summary.csv'
        header = ['size', 'size_nz', 'compression_ratio', 'flops', 'flops_nz', 'theoretical_speedup', 'loss',
                  'val_acc1', 'val_acc5']

        if not summary_path.exists():
            with open(summary_path, 'w') as f:
                f.write(','.join(header))
                f.write('\n')

        with open(summary_path, 'a') as f:
            values = [str(metrics[key]) for key in header]
            f.write(','.join(values))
            f.write('\n')

    def __repr__(self):
        if not isinstance(self.params['model'], str) and isinstance(self.params['model'], nn.Module):
            self.params['model'] = self.params['model'].__module__

        assert isinstance(self.params['model'], str), f"\nUnexpected model inputs: {self.params['model']}"
        return json.dumps(self.params, indent=4)

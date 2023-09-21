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

from datasets.transforms import get_mixup_cutmix
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
        'num_workers': 8
    }
    default_model_kwargs = {
        'pretrained': None
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
        self.trainer = pl.Trainer(default_root_dir=self.path, max_epochs=self.epochs,
                                  callbacks=[
                                      checkpoint_callback])  # callbacks=[checkpoint_callback, early_stop_callback])
        trainer = self.trainer

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

    def _build_dataloaders(self, dataset, **dl_kwargs):
        constructor = getattr(datasets, dataset)
        train_dataset, val_dataset, train_sampler, test_sampler = constructor()
        num_classes = len(dataset.classes)
        args = {'mixup_alpha': 0.2, 'cutmix_alpha': 1.0, 'use_v2': False}
        mixup_cutmix = get_mixup_cutmix(
            mixup_alpha=args['mixup_alpha'], cutmix_alpha=args['cutmix_alpha'], num_categories=num_classes,
            use_v2=args['use_v2']
        )
        if mixup_cutmix is not None:

            def collate_fn(batch):
                return mixup_cutmix(*default_collate(batch))

        else:
            collate_fn = default_collate

        # train_dataset = constructor(train=True)
        # val_dataset = constructor(train=False)
        # targets = dataset.targets
        # train_idx, val_idx = train_test_split(np.arange(len(targets)), test_size=dataset.val_size, random_state=42,
        #                                      shuffle=True, stratify=targets)
        # train_dataset = Subset(dataset, train_idx)
        # val_dataset = Subset(dataset, val_idx)
        self.train_dl = DataLoader(train_dataset, sampler=train_sampler, collate_fn=collate_fn, **dl_kwargs)
        self.val_dl = DataLoader(val_dataset, sampler=test_sampler, **dl_kwargs)

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

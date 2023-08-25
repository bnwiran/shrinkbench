import json
import logging
from typing import Union

from .train import TrainingExperiment
from .. import strategies
from ..metrics import model_size, flops


class PruningExperiment(TrainingExperiment):

    def __init__(self,
                 name: str,
                 dataset,
                 model,
                 strategy,
                 compression: Union[list[int], int],
                 resume_exp: bool = False,
                 seed=42,
                 path=None,
                 model_kwargs=None,
                 dl_kwargs=None,
                 train_kwargs=None,
                 debug=False,
                 resume=None,
                 resume_optim=False,
                 save_freq=10) -> None:

        super().__init__(name, dataset, model, resume_exp, seed, path, model_kwargs, dl_kwargs, train_kwargs, debug,
                         resume, resume_optim, save_freq)

        self.pruning = None
        self.strategy = strategy
        self.compression = [compression] if isinstance(compression, int) else compression
        self.add_params(strategy=strategy, compression=compression)
        self.save_freq = save_freq

    def setup(self):
        super().setup()

    def run(self):
        constructor = getattr(strategies, self.strategy)
        # self._save_metrics(0)

        for c in self.compression:
            logging.info(f'Running pruning experiment with compression {c}')
            x, y = next(iter(self.train_dl))

            self.pruning = constructor(self.model, x, y, compression=c)
            super().run()
            self.pruning.apply()
            self.run_epochs()
            self._save_metrics(c)

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

        val_metrics = self.trainer.test(self.model, self.val_dl)[0]
        metrics['loss'] = val_metrics['test_loss']
        metrics['val_acc1'] = val_metrics['test_acc1']
        metrics['val_acc5'] = val_metrics['test_acc5']

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

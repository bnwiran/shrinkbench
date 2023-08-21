import json
from typing import Union

from .train import TrainingExperiment

from .. import strategies
from ..metrics import model_size, flops
from ..util import printc


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
        for c in self.compression:
            super().run()
            x, y = next(iter(self.train_dl))
            self.pruning = constructor(self.model, x, y, compression=c)
            self.pruning.apply()
            self.save_metrics()

            if c >= 1:
                self.run_epochs()

    def save_metrics(self):
        metrics = self.pruning_metrics()
        with open(self.path / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        printc(json.dumps(metrics, indent=4), color='GRASS')
        summary = self.pruning.summary()
        summary_path = self.path / 'masks_summary.csv'
        summary.to_csv(summary_path)

    def pruning_metrics(self):
        metrics = {}
        # Model Size
        size, size_nz = model_size(self.model)
        metrics['size'] = size
        metrics['size_nz'] = size_nz
        metrics['compression_ratio'] = size / size_nz

        x, _ = next(iter(self.val_dl))
        x = x.to(self.model.device)

        # FLOPS
        ops, ops_nz = flops(self.model, x)
        metrics['flops'] = ops
        metrics['flops_nz'] = ops_nz
        metrics['theoretical_speedup'] = ops / ops_nz

        # Accuracy
        val_metrics = self.trainer.test(self.model, self.val_dl)[0]
        val_metrics['val_loss'] = val_metrics['test_loss']
        val_metrics['val_acc1'] = val_metrics['test_acc1']
        val_metrics['val_acc5'] = val_metrics['test_acc5']

        del val_metrics['test_loss']
        del val_metrics['test_acc1']
        del val_metrics['test_acc5']

        metrics['loss'] = val_metrics['val_loss']
        metrics['val_acc1'] = val_metrics['val_acc1']
        metrics['val_acc5'] = val_metrics['val_acc5']

        return metrics

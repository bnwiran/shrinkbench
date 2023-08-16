import json

from .train import TrainingExperiment, Trainer

from .. import strategies
from ..metrics import model_size, flops
from ..util import printc


class PruningExperiment(TrainingExperiment):

    def __init__(self,
                 dataset,
                 model,
                 strategy,
                 compression,
                 seed=42,
                 path=None,
                 dl_kwargs=dict(),
                 train_kwargs=dict(),
                 debug=False,
                 pretrained=True,
                 resume=None,
                 resume_optim=False,
                 save_freq=10):

        super(PruningExperiment, self).__init__(dataset, model, seed, path, dl_kwargs, train_kwargs, debug, pretrained, resume, resume_optim, save_freq)
        self.add_params(strategy=strategy, compression=compression)

        self.apply_pruning(strategy, compression)

        self.path = path
        self.save_freq = save_freq

    def apply_pruning(self, strategy, compression):
        constructor = getattr(strategies, strategy)
        x, y = next(iter(self.train_dl))
        self.pruning = constructor(self.model, x, y, compression=compression)
        self.pruning.apply()
        printc("Masked model", color='GREEN')

    def run(self):
        self.freeze()
        printc(f"Running {repr(self)}", color='YELLOW')
        self.to_device()
        self.build_logging(self.train_metrics, self.path)
        self.trainer = Trainer(self.model, self.path, self.optim, self.loss_func, self.save_freq, self.log)

        self.save_metrics()

        if self.pruning.compression > 1:
            self.run_epochs()

    def save_metrics(self):
        self.metrics = self.pruning_metrics()
        with open(self.path / 'metrics.json', 'w') as f:
            json.dump(self.metrics, f, indent=4)
        printc(json.dumps(self.metrics, indent=4), color='GRASS')
        summary = self.pruning.summary()
        summary_path = self.path / 'masks_summary.csv'
        summary.to_csv(summary_path)
        print(summary)

    def pruning_metrics(self):

        metrics = {}
        # Model Size
        size, size_nz = model_size(self.model)
        metrics['size'] = size
        metrics['size_nz'] = size_nz
        metrics['compression_ratio'] = size / size_nz

        x, y = next(iter(self.val_dl))
        x, y = x.to(self.device), y.to(self.device)

        # FLOPS
        ops, ops_nz = flops(self.model, x)
        metrics['flops'] = ops
        metrics['flops_nz'] = ops_nz
        metrics['theoretical_speedup'] = ops / ops_nz

        # Accuracy
        val_metrics = self.trainer.evaluate(self.val_dl)#run_epoch(False, -1)
        self.log(**val_metrics)
        self.log_epoch(0)

        metrics['loss'] = val_metrics['val_loss']
        metrics['val_acc1'] = val_metrics['val_acc1']
        metrics['val_acc5'] = val_metrics['val_acc5']

        return metrics

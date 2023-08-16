from abc import ABC, abstractmethod
import datetime
import hashlib
import json
import pathlib
import random
import shutil
import signal
import string
import sys

import numpy as np
import torch

from ..util import CSVLogger
from ..util import printc

import platform


class Experiment(ABC):

    def __init__(self, seed=42):
        self._params = {"experiment": self.__class__.__name__, 'params': {}}
        self.seed = seed
        self.frozen = False
        signal.signal(signal.SIGINT, self.SIGINT_handler)

        if platform.system() != 'Windows':
            signal.signal(signal.SIGQUIT, self.SIGQUIT_handler)

    def add_params(_self, **kwargs):
        if not _self.frozen:
            _self._params['params'].update({k: v for k, v in kwargs.items() if k not in ('self', '__class__')})
        else:
            raise RuntimeError("Cannot add params to frozen experiment")

    def freeze(self):
        self.generate_uid()
        self.fix_seed(self.seed)
        self.frozen = True

    @property
    def params(self):
        # prevents from trying to modify
        return self._params['params']

    def serializable_params(self):
        return {k: repr(v) for k, v in self._params.items()}

    def save_params(self):
        path = self.path / 'params.json'
        with open(path, 'w') as f:
            json.dump(self.serializable_params(), f, indent=4)

    def _get_path(self):
        if hasattr(self, "rootdir"):
            parent = pathlib.Path(self.rootdir)
        else:
            parent = pathlib.Path('results')
        if self._params.get('debug', False):
            parent /= 'tmp'
        parent.mkdir(parents=True, exist_ok=True)
        return parent / self.uid

    @property
    def digest(self):
        return hashlib.md5(json.dumps(self.serializable_params(), sort_keys=True).encode('utf-8')).hexdigest()

    def __hash__(self):
        return hash(self.digest)

    def fix_seed(self, seed=42, deterministic=False):
        # https://pytorch.org/docs/stable/notes/randomness.html

        # Python
        random.seed(seed)

        # Numpy
        np.random.seed(seed)

        # PyTorch
        torch.manual_seed(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def generate_uid(self):
        """Returns a time sortable UID

        Computes timestamp and appends unique identifie

        Returns:
            str -- uid
        """
        if hasattr(self, "uid"):
            return self.uid

        N = 4  # length of nonce
        time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        nonce = ''.join(random.choices(string.ascii_uppercase + string.digits, k=N))
        self.uid = f"{time}-{nonce}-{self.digest}"
        return self.uid

    def _build_logging(self, metrics, path=None, log_csv=True, log_tb=False):
        if path is None:
            self.path = self._get_path()
        printc(f"Logging results to {self.path}", color='MAGENTA')
        self.path.mkdir(exist_ok=True, parents=True)
        self.save_params()

        self.log_csv = log_csv
        self.log_tb = log_tb
        self.log_epoch_n = 0
        if self.log_csv:
            self.csv_logger = CSVLogger(self.path / 'logs.csv', metrics)
        if self.log_tb:
            tb_path = self.path / 'tbevents'
            tb_path.mkdir()
            from torch.utils.tensorboard import SummaryWriter
            self.tb_logger = SummaryWriter(log_dir=tb_path)

    def log(self, **kwargs):
        if self.log_csv:
            self.csv_logger.set(**kwargs)
        if self.log_tb:
            for k, v in kwargs.items():
                self.tb_writer.add_scalar(k, v, self.log_epoch_n)

    def log_epoch(self, epoch=None):
        if epoch is not None:
            self.log_epoch_n = epoch
        self.log_epoch_n += 1
        if self.log_csv:
            self.csv_logger.set(epoch=epoch)
            self.csv_logger.update()
            self.csv_logger.set(epoch=self.log_epoch_n)

    def SIGINT_handler(self, sig, frame):
        pass

    def SIGQUIT_handler(self, sig, frame):
        shutil.rmtree(self.path, ignore_errors=True)
        self.wrapup()
        sys.exit(1)

    def start(self):
        self.setup()
        self.run()

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def wrapup(self):
        self.csv_logger.close()

    def __repr__(self):
        return json.dumps(self.params, indent=4)

import json
import platform
import random
import shutil
import signal
import sys
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch

from ..util import printc


class Experiment(ABC):

    def __init__(self, name: str, path: str, resume: bool = False, seed=42) -> None:
        assert name and name.strip(), "Experiment must have a name"

        self.path = self._create_experiment_dir(path, name, resume)
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

    def _create_experiment_dir(self, path: str, name: str, resume: bool = False) -> Path:
        if path is None:
            path = '.'

        abs_path = Path(path + "/" + name)
        if abs_path.is_dir() and not resume:
            raise Exception(f"Cannot create new experiment with name {name} on {path}, it's already exists")

        abs_path.mkdir(parents=True, exist_ok=True)

        return abs_path

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

    def _build_logging(self):
        printc(f"Logging results to {self.path}", color='MAGENTA')
        self.save_params()

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
        pass

    def __repr__(self):
        return json.dumps(self.params, indent=4)

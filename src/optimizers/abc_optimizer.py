from abc import ABC, abstractmethod

from src.data_loaders import ABCLoader, BaseLoader
from src.layers import ABCLayer
from src.regularizers import ABCRegularizer


class ABCOptimizer(ABC):
    def __init__(
            self,
            model_weights_layers: list[ABCLayer],
            data_loader: ABCLoader | None = None,
            lr: float = 0.001
        ):
        self.model_weights_layers = model_weights_layers
        self.data_loader = BaseLoader() if data_loader is None else data_loader
        self.lr = lr

    @abstractmethod
    def step(self, regularizer: ABCRegularizer | None = None):
        raise NotImplementedError()

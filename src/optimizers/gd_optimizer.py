import numpy as np

from src.optimizers import ABCOptimizer
from src.regularizers import ABCRegularizer
from src.data_loaders import ABCLoader
from src.layers import ABCLayer


class GDOptimizer(ABCOptimizer):
    def __init__(
            self,
            model_weights_layers: list[ABCLayer],
            data_loader: ABCLoader | None = None,
            lr: float = 0.001
        ):
        super().__init__(model_weights_layers, data_loader, lr)

    def step(self, regularizer: ABCRegularizer | None = None):
        for layer in reversed(self.model_weights_layers):
            weights: np.ndarray = layer.get_weights()
            gradients: np.ndarray = layer.get_gradients()
            weights -= regularizer.pd_wrt_w(self.lr, weights, gradients) if regularizer else self.lr*gradients
            layer.update_weights(weights)

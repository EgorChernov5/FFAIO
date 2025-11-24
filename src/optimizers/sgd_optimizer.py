import numpy as np

from src.optimizers import Optimizer
from src.data_loaders import ShuffleLoader
from src.layers import Layer
from src.losses import Loss


class SGD(Optimizer):
    def __init__(self, batch_size: int, lr: float):
        super().__init__()

        self.batch_size = batch_size
        self.lr = lr

        self.data_loader = ShuffleLoader(batch_size)

    def step(self):
        prev_layer = None
        for layer in reversed(self.arch_model):
            weights = layer.get_weights()
            grad_layer = layer.gradient(self.loss, prev_layer)
            weights -= self.lr*grad_layer
            layer.update_weights(weights)
            prev_layer = layer

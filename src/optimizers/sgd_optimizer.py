import numpy as np

from src.optimizers import Optimizer
from src.data_loaders import ShuffleLoader
from src.layers import Layer
from src.losses import Loss


class SGD(Optimizer):
    def __init__(self, batch_size: int, lr: float, random_state: int | None = None):
        super().__init__()

        self.batch_size = batch_size
        self.lr = lr

        self.data_loader = ShuffleLoader(batch_size, random_state)

    def step(self):
        # Берём выходной слой и его веса
        prev_layer = self.arch_model[-1]
        prev_weights = prev_layer.get_weights()
        # Считаем градиент выходного слоя
        prev_grad_layer = prev_layer.gradient(self.loss)
        # Обновление весов для скрытых слоёв
        for layer in reversed(self.arch_model[:-1]):
            # Извлекаем веса и считаем градиент скрытого слоя
            weights = layer.get_weights()
            grad_layer = layer.gradient(self.loss, prev_layer)
            # Обновляем веса предыдущего слоя
            prev_weights -= self.lr*prev_grad_layer
            prev_layer.update_weights(prev_weights)
            # Перезаписываем переменные
            prev_layer = layer
            prev_weights = weights
            prev_grad_layer = grad_layer

        # Обновляем веса входного слоя
        prev_weights -= self.lr*prev_grad_layer
        prev_layer.update_weights(prev_weights)

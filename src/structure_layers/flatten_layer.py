import numpy as np

from src.structure_layers import ABCStructureLayer


class FlattenLayer(ABCStructureLayer):
    def __init__(self):
        super().__init__()
        self.original_shape = None  # чтобы помнить форму входа

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward pass.

        :param inputs: Тензор формы (batch, C, H, W) или любой другой формы. 
        :type inputs: np.ndarray

        :return: Возвращает плоский тензор (batch, -1). 
        :rtype: np.ndarray
        """
        # Сохраняем исходную форму
        self.original_shape = inputs.shape

        # Выпрямляем начиная со 2-й оси (0-я - batch)
        batch_size = inputs.shape[0]
        return inputs.reshape(batch_size, -1)

    def backward_pass(self, delta: np.ndarray) -> np.ndarray:
        """
        Backward pass.

        :param delta: Градиенты со следующего слоя, формы (batch, flattened_size)
        :type delta: np.ndarray

        :return: Градиенты, восстановленные в исходную форму входа
        :rtype: np.ndarray
        """
        if self.original_shape is None:
            raise ValueError("Forward pass must be called before backward_pass.")

        # Reshape обратно в исходную форму входных данных
        return delta.reshape(self.original_shape)

import numpy as np

from src.structure_layers import ABCStructureLayer


class DropoutLayer(ABCStructureLayer):
    def __init__(self, dropout_rate: float = 0.5):
        """
        :param dropout_rate: Вероятность "выключения" нейронов (0.0-1.0)
        """
        super().__init__()
        if not 0 <= dropout_rate < 1:
            raise ValueError("dropout_rate must be in [0, 1).")
        self.dropout_rate = dropout_rate
        self.mask = None
        self.learning = True  # По умолчанию слой в режиме обучения

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward pass.
        :param inputs: Входные данные формы (batch, ...).
        :return: Входные данные с применённым dropout.
        """
        if self.learning:
            # Генерируем маску с 0/1
            self.mask = (np.random.rand(*inputs.shape) >= self.dropout_rate).astype(inputs.dtype)
            # Масштабируем оставшиеся элементы, чтобы сохранить математическое ожидание
            return inputs * self.mask / (1.0 - self.dropout_rate)
        else:
            # Во время инференса dropout не применяется
            return inputs

    def backward_pass(self, delta: np.ndarray) -> np.ndarray:
        """
        Backward pass.
        :param delta: Градиенты со следующего слоя, той же формы, что и inputs.
        :return: Градиенты после применения dropout mask.
        """
        if self.learning and self.mask is not None:
            return delta * self.mask / (1.0 - self.dropout_rate)
            
        return delta
    
    def train(self):
        self.learning = True

    def eval(self):
        self.learning = False

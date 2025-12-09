import numpy as np
from abc import ABC, abstractmethod


class ABCLoss(ABC):
    def __init__(self):
        self.learning: bool = True

        self.y_true: np.ndarray | None = None

    @abstractmethod
    def __call__(self, y_true: np.ndarray, logits: np.ndarray) -> float:
        """
        Подсчёт ошибки на истинных значениях и предсказанных.

        :param y_true: Истинные метки класса.
        :type y_true: np.ndarray
        :param logits: Значения с последнего слоя.
        :type logits: np.ndarray

        :return: Значение ошибки на объектах.
        :rtype: float
        """
        raise NotImplementedError()
    
    def to_str(self):
        return str(self).split('.')[-1].split()[0]
    
    @abstractmethod
    def backward_pass(self, logits: np.ndarray) -> np.ndarray:
        """
        Частная производная функции ошибки по входам.

        :param logits: Значения с последнего слоя.
        :type logits: np.ndarray

        :return: Приращение функции ошибки по входам.
        :rtype: np.ndarray
        """
        raise NotImplementedError()
    
    def train(self):
        self.learning = True

    def eval(self):
        self.learning = False

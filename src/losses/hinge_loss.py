import numpy as np

from src.losses import ABCLoss


class HingeLoss(ABCLoss):
    """
    Шарнирная функция ошибки.
    """
    def __call__(self, y_true: np.ndarray, logits: np.ndarray) -> np.float64:
        if self.learning: self.y_true = y_true.copy()
        return np.mean(np.maximum(0, 1 - y_true*logits))
    
    def backward_pass(self, logits: np.ndarray) -> tuple[np.ndarray, np.float64]:
        return -np.mean(logits*self.y_true, axis=0), -np.mean(self.y_true)

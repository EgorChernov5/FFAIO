import numpy as np

from src.losses import ABCLoss


class HingeLoss(ABCLoss):
    """
    Шарнирная функция ошибки.
    """
    def __call__(self, y_true: np.ndarray, Z: np.ndarray) -> np.float64:
        if self.learning: self.y_true = y_true.copy()
        return np.mean(np.maximum(0, 1 - y_true*Z))
    
    def backward_pass(self, y_pred: np.ndarray) -> tuple[np.ndarray, np.float64]:
        return -np.mean(y_pred*self.y_true, axis=0), -np.mean(self.y_true)

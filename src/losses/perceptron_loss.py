import numpy as np

from src.losses import ABCLoss


class PerceptronLoss(ABCLoss):
    """
    Функция ошибки персептрона или кусочно-линейная функция потерь (Hebb’s rule).
    """
    def __call__(self, y_true: np.ndarray, logits: np.ndarray) -> float:
        if self.learning: self.y_true = y_true.copy()
        return np.mean(np.maximum(0, -y_true*logits))
    
    def backward_pass(self, logits: np.ndarray) -> np.ndarray:
        return np.where(self.y_true*logits < 0, -self.y_true, 0.)

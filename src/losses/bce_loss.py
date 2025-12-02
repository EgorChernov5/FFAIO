import numpy as np

from src.losses import ABCLoss
from src.tools import utils


class BCELoss(ABCLoss):
    def __init__(self, eps: float = 1e-7):
        super().__init__()
        
        self.eps = eps

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        assert y_true.ndim != 1, f'Пространство меток должно быть 2, а не {y_true.ndim}'
        assert y_true.shape == y_pred.shape, f'Истинные значения и выходы модели должны быть одной формы {y_true.shape} != {y_pred.shape}'
        if self.learning: self.y_true = y_true.copy()

        y_pred = np.clip(y_pred, self.eps, 1 - self.eps)

        if y_true.shape[1] == 1:
            loss = -y_true*np.log(y_pred) - (1 - y_true)*np.log(1 - y_pred)
        else:
            loss = -np.sum(y_true*np.log(y_pred), axis=1)

        return np.mean(loss)
    
    def backward_pass(self, y_pred: np.ndarray) -> np.ndarray:
        n_batch = y_pred.shape[0]
        y_pred = np.clip(y_pred, self.eps, 1 - self.eps)

        if self.y_true.shape[1] == 1:
            dL = (-self.y_true/y_pred) + ((1 - self.y_true)/(1 - y_pred))
        else:
            dL = -self.y_true/y_pred

        return dL/n_batch

import numpy as np

from src.regularizers import ABCRegularizer


class L2Regularizer(ABCRegularizer):
    def __init__(self, lambda_q: float = 0.1):
        super().__init__(lambda_q)

    def __call__(self, weights: np.ndarray, losses: np.ndarray) -> np.ndarray:
        return losses + self.lambda_q*np.sum(np.square(weights))
    
    def pd_wrt_w(self, lr: float, weights: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        return lr*self.lambda_q*weights + lr*gradients

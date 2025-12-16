import numpy as np

from src.regularizers import ABCRegularizer


class L1Regularizer(ABCRegularizer):
    def __init__(self, lambda_q: float = 0.1, bias_regularizer: bool = False):
        super().__init__(lambda_q, bias_regularizer)

    def __call__(self, weights: dict) -> np.ndarray:
        sum_weights = 0.0
        for name, w in weights.items():
            if 'bias' in name:
                if self.bias_regularizer:
                    sum_weights += np.sum(np.abs(w))
            else:
                sum_weights += np.sum(np.abs(w))
        return self.lambda_q * sum_weights

    def pd_wrt_w(self, weights: tuple[np.ndarray, np.ndarray | None]) -> tuple[np.ndarray, np.ndarray | None]:
        W, b = weights
        reg_W = self.lambda_q * np.sign(W)
        reg_b = self.lambda_q * np.sign(b) if self.bias_regularizer and b is not None else None
        return reg_W, reg_b

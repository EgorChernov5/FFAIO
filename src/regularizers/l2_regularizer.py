import numpy as np

from src.regularizers import ABCRegularizer


class L2Regularizer(ABCRegularizer):
    def __init__(self, lambda_q: float = 0.1, bias_regularizer: bool = False):
        super().__init__(lambda_q, bias_regularizer)

    def __call__(self, weights: dict) -> np.ndarray:
        for name_layer, weights in weights.items():
            if 'bias' in name_layer and self.bias_regularizer:
                sum_weights += np.sum(np.square(weights))
            elif 'bias' not in name_layer:
                sum_weights += np.sum(np.square(weights))
        # W, b = weights
        # sum_weights = np.sum(np.square(W)) + np.sum(np.square(b)) if self.bias_regularizer else np.sum(np.square(W))
        return 0.5*self.lambda_q*sum_weights
    
    def pd_wrt_w(
        self,
        weights: tuple[np.ndarray, np.ndarray | None]
        ) -> tuple[np.ndarray, np.ndarray | None]:
        W, b = weights
        if self.bias_regularizer:
            return self.lambda_q*W, self.lambda_q*b
        else:
            return self.lambda_q*W, None

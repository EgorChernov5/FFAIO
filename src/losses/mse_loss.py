import numpy as np

from src.losses import Loss


class MSE(Loss):
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        self.losses = np.array([np.mean((y_true - y_pred)**2)])
        return self.losses
    
    def partial_derivative_wrt_a(self, i_neuron: int, input: float) -> float:
        n = len(self.losses)
        return 2/n*(input - self.losses[i_neuron])

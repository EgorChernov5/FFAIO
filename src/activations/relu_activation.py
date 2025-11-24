import numpy as np

from src.activations import ActivationFunction


class ReLU(ActivationFunction):
    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        features = np.maximum(0, inputs)
        # Усреднение по выходам в батче
        self.A = np.mean(features, axis=1)
        return features
    
    # def partial_derivative_wrt_z(self, input: float) -> float:
    #     return 1. if np.maximum(0, input) else 0.
    
    def partial_derivative_wrt_z(self, i_neuron: int) -> float:
        return 1. if self.A[i_neuron] > 0. else 0.

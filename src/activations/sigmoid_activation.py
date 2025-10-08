import numpy as np

from src.activations import ActivationFunction


class Sigmoid(ActivationFunction):
    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        # Расчёт выходов активации для нейронов на слое
        self.A = 1 / (1 + np.exp(-inputs))
        return self.A
    
    def partial_derivative_wrt_z(self, i_neuron: int) -> np.ndarray:
        return self.A[:, i_neuron]*(1 - self.A[:, i_neuron])

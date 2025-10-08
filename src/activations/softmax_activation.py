import numpy as np

from src.activations import ActivationFunction



class Softmax(ActivationFunction):
    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        exp = np.exp(inputs - np.max(inputs))
        self.A = exp/np.sum(exp)
        return self.A
    
    def partial_derivative_wrt_z(self, i_neuron: int) -> float:
        # Матрица Якоби
        jacobian = np.diagflat(self.A) - np.outer(self.A, self.A)
        return jacobian

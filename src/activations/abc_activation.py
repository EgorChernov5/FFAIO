import numpy as np
from abc import ABC, abstractmethod


class ActivationFunction(ABC):
    def __init__(self):
        self.A: np.ndarray | None = None
        self.dA_dZ: np.ndarray | None = None
        self.dA_dZ_p: np.ndarray | None = None

    @abstractmethod
    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def partial_derivative_wrt_z(self, i_neuron: int) -> np.ndarray:
        raise NotImplementedError()

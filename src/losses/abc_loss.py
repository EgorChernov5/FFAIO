import numpy as np
from abc import ABC, abstractmethod


class Loss(ABC):
    def __init__(self):
        self.ty_true: np.ndarray | None = None
        self.losses: np.ndarray | None = None
        self.dL_dA: np.ndarray | None = None
        self.dL_dA_p: np.ndarray | None = None
    
    @abstractmethod
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        raise NotImplementedError()
    
    # TODO: возвращать np.ndarray
    # @abstractmethod
    # def partial_derivative_wrt_a(self, i_neuron: int, input: float) -> float:
    #     raise NotImplementedError()
    @abstractmethod
    def partial_derivative_wrt_a(self, i_neuron: int, input: float) -> np.ndarray:
        raise NotImplementedError()
    
    def to_str(self):
        return str(self).split('.')[-1].split()[0]

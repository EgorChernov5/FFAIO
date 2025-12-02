import numpy as np
from abc import ABC, abstractmethod


class ABCStructureLayer(ABC):
    def __init__(self):
        self.inputs: np.ndarray | None = None 
        self.outputs: np.ndarray | None = None

    @abstractmethod
    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def backward_pass(self, delta: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

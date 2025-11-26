import numpy as np

from src.activations import ABCActivation


class ReLUActivation(ABCActivation):
    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        if self.learning: self.inputs = inputs.copy()
        self.outputs = np.maximum(0, inputs)
        return self.outputs
    
    def pd_wrt_inputs(self) -> float:
        return np.where(self.outputs > 0., 1., 0.)

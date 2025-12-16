import numpy as np

from src.activations import ABCActivation


class TanhActivation(ABCActivation):
    def __init__(self):
        super().__init__()

        self.outputs_list = []   # tanh(Z_t)

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        # inputs: Z_t (batch, hidden)
        outputs = np.tanh(inputs)

        if self.learning:
            self.inputs = inputs.copy()
            self.outputs = outputs.copy()
            self.outputs_list.append(outputs.copy())

        return outputs

    def reset(self):
        """Call this at start of each forward pass."""
        self.outputs_list = []

    def backward_pass(self, delta: np.ndarray, t: int | None = None) -> np.ndarray:
        """
        delta_list: list[delta_h_t], each shape (batch, hidden)
        returns: list[dZ_t], same shapes
        """
        if t is not None:
            dA = 1 - self.outputs_list[t]**2
        else:
            dA = 1 - self.outputs**2
        
        return delta*dA

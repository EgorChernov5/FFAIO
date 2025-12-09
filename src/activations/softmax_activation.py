import numpy as np

from src.activations import ABCActivation


class SoftmaxActivation(ABCActivation):
    def __init__(self, is_log: bool = False):
        super().__init__()
        self.is_log = is_log
        
        self.learning = True
        
        self.inputs = None
        self.probs = None
        self.outputs = None

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        # Вычисляем softmax с численной стабильностью
        shifted: np.ndarray = inputs - np.max(inputs, axis=-1, keepdims=True)
        probs: np.ndarray = np.exp(shifted) / np.sum(np.exp(shifted), axis=-1, keepdims=True)
        outputs = probs.copy()
        # Если надо, то считаем log_softmax
        if self.is_log:
            outputs = np.log(outputs)
        
        if self.learning:
            self.inputs = inputs.copy()
            self.probs = probs.copy()
            self.outputs = outputs.copy()
        
        return outputs

    def backward_pass(self, delta: np.ndarray) -> np.ndarray:
        """
        Производная вычисляется в функции CCELoss. Поэтому мы просто передаем результат дальше.
        """
        return delta

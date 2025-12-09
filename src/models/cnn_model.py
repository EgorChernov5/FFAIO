import numpy as np

from src.models import ABCModel
from src.layers import ABCLayer
from src.activations import ABCActivation
from src.structure_layers import ABCStructureLayer, FlattenLayer


class CNN(ABCModel):
    def __init__(self, arch_model: list[ABCLayer| ABCActivation | ABCStructureLayer]):
        super().__init__(arch_model)

    def get_features(self, X: np.ndarray) -> np.ndarray:
        outputs = X
        for layer in self.arch_model:
            outputs = layer(outputs)
            if isinstance(layer, FlattenLayer):
                break

        return outputs

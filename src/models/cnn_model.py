from src.models import ABCModel
from src.layers import ABCLayer
from src.activations import ABCActivation
from src.structure_layers import ABCStructureLayer


class CNN(ABCModel):
    def __init__(self, arch_model: list[ABCLayer| ABCActivation | ABCStructureLayer]):
        super().__init__(arch_model)

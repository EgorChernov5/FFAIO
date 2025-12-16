from src.models import ABCModel
from src.layers import ABCLayer
from src.activations import ABCActivation
from src.structure_layers import ABCStructureLayer
from src.cells import ABCCell


class LSTM(ABCModel):
    def __init__(self, arch_model: list[ABCLayer| ABCActivation | ABCStructureLayer | ABCCell]):
        super().__init__(arch_model)

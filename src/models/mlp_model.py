from src.models import ABCModel
from src.layers import ABCLayer
from src.activations import ABCActivation


class MLP(ABCModel):
    def __init__(self, arch_model: list[ABCLayer| ABCActivation]):
        super().__init__(arch_model)

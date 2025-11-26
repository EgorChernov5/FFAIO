from src.models import ABCModel
from src.layers import LinearLayer
from src.activations import SignActivation


class LinearClassificator(ABCModel):
    def __init__(self, in_features: int):
        super().__init__([
            LinearLayer(in_features, 1, True),
            SignActivation()
        ])

from src.models import ABCModel


class CNN(ABCModel):
    def __init__(self, arch_model):
        super().__init__(arch_model)

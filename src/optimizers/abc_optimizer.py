from abc import ABC, abstractmethod

from src.layers import Layer
from src.losses import Loss


class Optimizer(ABC):
    def __init__(self):
        self.arch_model: list[Layer] | None = None
        self.loss: Loss | None = None

    def add_params(self, arch_model: list[Layer], loss: Loss) -> None:
        self.arch_model = arch_model
        self.loss = loss

    @abstractmethod
    def step(self) -> None:
        raise NotImplementedError()

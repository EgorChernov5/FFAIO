from abc import ABC
from typing import Iterator
import numpy as np


class ABCLoader(ABC):
    def __init__(self):
        pass

    def get_data(self, X: np.ndarray, y: np.ndarray) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
        raise NotImplementedError()

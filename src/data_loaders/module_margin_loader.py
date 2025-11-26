from typing import Iterator
import numpy as np

from src.data_loaders import ABCLoader


class ModuleMarginLoader(ABCLoader):
    def get_data(self, X: np.ndarray, y: np.ndarray) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
        pass
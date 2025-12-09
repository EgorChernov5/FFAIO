from typing import Iterator
import numpy as np

from src.data_loaders import ABCLoader


class ShuffleLoader(ABCLoader):
    def __init__(self, batch_size: int = 32):
        super().__init__()

        self.batch_size = batch_size

    def get_data(self, X: np.ndarray, y: np.ndarray) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
        n_samples = len(y)
        # Перемешиваем индексы один раз
        inds = np.random.permutation(n_samples)
        for n_batch, start in enumerate(range(0, n_samples, self.batch_size)):
            end = min(start + self.batch_size, n_samples)
            batch_inds = inds[start:end]
            yield n_batch, X[batch_inds], y[batch_inds]

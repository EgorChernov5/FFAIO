import numpy as np


def to_one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    """
    Преобразует вектор меток классов в one-hot encoding.

    :param y: Вектор меток классов размерности (n_batch,), значения от 0 до n_classes-1.
    :type y: np.ndarray
    :param n_classes: Общее количество классов.
    :type n_classes: int

    :return: One-hot представление меток размерности (n_batch, n_classes).
    :rtype: np.ndarray
    """
    y = np.asarray(y, dtype=int)
    one_hot = np.zeros((y.shape[0], n_classes), dtype=np.float32)
    one_hot[np.arange(y.shape[0]), y] = 1.0
    return one_hot

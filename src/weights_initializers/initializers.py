import numpy as np
import torch
from torch.nn import init


def correlated_init():
    pass


def random_numbers_init(n_in, sigma=None):
    """
    Эвристическая инициализация весов случайными числами из Var(w).
    
    :param n_in: Количество входов слоя.
    :type n_in: int
    :param sigma: Стандартное отклонение (если None, можно выбрать 1/√n_in).
    :type sigma: float | None
    
    :return: W — матрица весов размерности (n_out, n_in).
    :rtype: np.ndarray
    """
    if sigma is None:
        sigma = 1.0 / np.sqrt(n_in)
    
    W = np.random.normal(0.0, sigma, n_in)
    return W


def xavier_init(n_in, n_out):
    """
    Инициализация весов по методу Ксавье.

    :param n_in: Количество входов слоя.
    :param n_out: Количество выходов слоя.

    :return: W — матрица весов размерности (n_out, n_in).
    :rtype: np.ndarray
    """
    limit = np.sqrt(6.0 / (n_in + n_out))
    W = np.random.uniform(-limit, limit, n_in)
    return W


def he_init(n_in):
    """
    Инициализация весов по методу He.
    
    :param n_in: Количество входов слоя.
    :type n_in: int

    :return: W — матрица весов размерности (n_out, n_in).
    :rtype: np.ndarray
    """
    stddev = np.sqrt(2.0 / n_in)
    W = np.random.normal(0.0, stddev, n_in)
    return W


def conv_uniform_init(shape: list[int], bias: bool = True) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Docstring for conv_uniform_init
    
    :param shape: Description (shape: [out_channels, in_channels, kernel_size[0], kernel_size[1]])
    :type shape: list[int]
    :param bias: Description
    :type bias: bool

    :return: Description
    :rtype: tuple[ndarray[_AnyShape, dtype[Any]], ndarray[_AnyShape, dtype[Any]] | None]
    """
    assert len(shape) == 4, f'Dimention of inputs should be 4, but got {len(shape)}'
    
    k = 1/(shape[1]*shape[2]*shape[3])
    W = init.uniform_(torch.zeros(shape), a=-np.sqrt(k), b=np.sqrt(k)).cpu().numpy()
    b = init.uniform_(torch.zeros([shape[0]]), a=-np.sqrt(k), b=np.sqrt(k)).cpu().numpy() if bias else None
    return W, b

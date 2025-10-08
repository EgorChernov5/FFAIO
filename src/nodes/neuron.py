import numpy as np


class Neuron:
    def __init__(self, in_features: int, bias: bool):
        self.in_features = in_features

        self.weights = np.random.rand(self.in_features)
        self.bias = np.random.rand(1)[0] if bias else np.zeros(1)[0]

        self.X: np.ndarray | None = None
        # self.Z = 0.
        self.Z: np.ndarray | None = None

    def len_weights(self):
        return len(self.weights) + 1 if self.bias else len(self.weights)
    
    def get_weights(self) -> np.ndarray:
        return np.append(self.weights, self.bias) if self.bias else self.weights
    
    def update_weights(self, weights: np.ndarray):
        # Валидируем веса
        error = self._check_size(weights)
        assert len(error) == 0, error

        # Обновляем веса
        if self.bias and (len(weights) - 1) == len(self.weights):
            self.weights, self.bias = weights[:-1], weights[-1]
        else:
            self.weights = weights

    def __call__(self, inputs: np.ndarray) -> float:
        """
        Пропускаем данные через нейрон.

        :param inputs: Набор объектов с размером [batch_size, n_features]
        :type inputs: np.ndarray

        :return: Выход нейрона после применения линейного преобразования.
        :rtype: float
        """
        self.X = inputs.copy()
        self.Z = self.X@self.weights + self.bias if self.bias else self.X@self.weights
        return self.Z

    # TODO
    # def partial_derivative_wrt_w(self, i_weight: int) -> float:
    #     return 1. if self.bias and i_weight == len(self.weights) else np.mean(self.X[:, i_weight])
    def partial_derivative_wrt_w(self, i_weight: int) -> np.ndarray:
        return np.ones(len(self.X)) if self.bias and i_weight == len(self.weights) else self.X[:, i_weight]
    
    def partial_derivative_wrt_a(self, i_feature: int) -> float:
        return self.weights[i_feature]
    
    def _check_size(self, weights) -> str:
        error = ''
        if self.bias:
            if (len(weights) - 1) != len(self.weights):
                error = 'Используется смещение, но значение отстуствует'
        else:
            if len(weights) != len(self.weights):
                error = 'Не совпадают размеры весов'

        return error

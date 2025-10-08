import numpy as np

from src.losses import Loss


class BinaryCrossEntropy(Loss):
    def __init__(self, labels: dict[int, str], eps: float = 1e-7):
        super().__init__()

        self.labels = labels
        self.eps = eps

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Вычисление ошибки.

        :param y_true: Список с истинными метками классов (размер: [n_samples,]).
        :type y_true: np.ndarray
        :param y_pred: Список с распределениями вероятностей отнесения к соответствующему классу объектов (размер: [n_samples, n_labels]).
        :type y_pred: np.ndarray

        :return: Cross-entropy ошибка.
        :rtype: float
        """
        # Трансформируем y_true с помощью one-hot кодирования
        n_labels = len(self.labels)
        ty_true = self._to_one_hot(y_true, n_labels)
        self.y = ty_true
        # Если бинарная классификация, то подгоняем y_pred под формулу кросс-энтропии в общем виде
        ty_pred = y_pred.copy()
        if len(self.labels) == 2:
            ty_pred = self._expand_binary_probs(ty_pred)
        
        # Добавляем маленькое число, чтобы исключить взятие log от 0
        ty_pred = np.clip(ty_pred, self.eps, None)
        # Считаем ошибку для каждого класса
        losses = -ty_true*np.log(ty_pred)
        # Усредняем ошибки по классам
        self.losses = np.mean(losses, axis=0)
        # Если бинарная классификация и выходной слой отдаёт 1 значение, то подгоняем формат
        if len(self.labels) == 2 and (y_pred.ndim == 1 or y_pred.shape[1] == 1):
            self.losses = np.array([np.mean(self.losses)])

        return np.mean(self.losses)
    
    # TODO
    # def partial_derivative_wrt_a(self, i_neuron: int, inputs: np.ndarray) -> float:
    #     t_inputs = self._expand_binary_probs(inputs)
    #     return -np.mean(self.y[:, i_neuron]/t_inputs[:, i_neuron])
    def partial_derivative_wrt_a(self, i_neuron: int, inputs: np.ndarray) -> np.ndarray:
        t_inputs = self._expand_binary_probs(inputs)
        return -self.y[:, i_neuron]/t_inputs[:, i_neuron]

    def _to_one_hot(self, y_true: np.ndarray, n_labels: int | None = None) -> np.ndarray:
        """
        Преобразует список с метками в one-hot encoding список.

        :param y_true: Массив меток классов shape = [n_samples,].
        :type y_true: np.ndarray
        :param n_labels: Количество классов (если None, берётся max + 1).
        :type n_labels: int | None

        :return: Список закодированными метками класса размера [n_samples, n_labels]
        :rtype: np.ndarray
        """
        if n_labels is None:
            n_labels = np.max(y_true) + 1
        
        return np.eye(n_labels)[y_true].squeeze()
    
    def _expand_binary_probs(self, y_pred: np.ndarray) -> np.ndarray:
        """
        Преобразует список вероятностей одного класса [n_samples, 1]
        в вероятности двух классов [n_samples, 2].

        :param y_pred: массив вероятностей первого класса (размер: [n_samples, 1]).
        :type y_pred: np.ndarray

        :return: Массив вероятностей [p, 1 - p] (размер: [n_samples, 2]).
        :rtype: np.ndarray
        """
        # Если размер [n_samples,], то переводим в [n_samples, 1]
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)

        # Если и так уже две вероятности, то менять ничего не надо
        if y_pred.shape[1] == 2:
            return y_pred

        p = y_pred[:, 0]
        return np.column_stack([1 - p, p])

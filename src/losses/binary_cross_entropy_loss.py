import numpy as np

from src.losses import Loss


class BinaryCrossEntropy(Loss):
    def __init__(self, labels: dict[int, str], eps: float = 1e-7):
        super().__init__()

        self.n_labels = len(labels)
        self.eps = eps

        self.n_output_neurons = None

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
        # 1. Предобработка данных
        # Трансформируем y_true с помощью one-hot кодирования
        ty_true = self._to_one_hot(y_true)
        # Если бинарная классификация, то подгоняем y_pred под формулу кросс-энтропии в общем виде
        self.n_output_neurons = 1 if self.n_labels == 2 and (y_pred.ndim == 1 or y_pred.shape[1] == 1) else self.n_labels
        ty_pred = y_pred.copy()
        if self.n_output_neurons == 1:
            ty_pred = self._expand_binary_probs(ty_pred)        
        
        # 2. Расчитываем потерю для классов
        # Считаем ошибку для каждого класса (добавляем eps, чтобы исключить log от 0)
        losses = -ty_true*np.log(ty_pred + self.eps)
        # Усредняем ошибки по классам
        losses = np.mean(losses, axis=0)
        # Если бинарная классификация и выходной слой отдаёт 1 значение, то подгоняем формат
        if self.n_output_neurons == 1:
            losses = np.array([np.mean(losses)])

        # 3. Сохранение данных
        # Сохраняем необходимые данные и инициализируем хранение производной
        self.ty_true = ty_true
        self.losses = losses
        self.dL_dA = np.empty((0, len(y_true)))

        return np.mean(self.losses)
    
    def partial_derivative_wrt_a(self, i_neuron: int, A: np.ndarray) -> np.ndarray:
        # Если бинарная классификация и выходной слой отдаёт 1 значение, то подгоняем формат
        if self.n_output_neurons == 1:
            y = self.ty_true[:, 1]
            a = np.clip(A.squeeze(), self.eps, 1 - self.eps)
        else:
            y = self.ty_true[:, i_neuron]
            a = np.clip(A[:, i_neuron], self.eps, 1 - self.eps)
        
        dL_da = -(y / a) + ((1 - y) / (1 - a))
        self.dL_dA = np.vstack((self.dL_dA, dL_da))
        return dL_da

    def _to_one_hot(self, y_true: np.ndarray) -> np.ndarray:
        """
        Преобразует список с метками в one-hot encoding список.

        :param y_true: Массив меток классов shape = [n_samples,].
        :type y_true: np.ndarray

        :return: Список закодированными метками класса размера [n_samples, n_labels]
        :rtype: np.ndarray
        """
        return np.eye(self.n_labels)[y_true].squeeze()
    
    def _expand_binary_probs(self, y_pred: np.ndarray) -> np.ndarray:
        """
        Преобразует список вероятностей одного класса [n_samples, 1]
        в вероятности двух классов [n_samples, 2].

        :param y_pred: массив вероятностей первого класса (размер: [n_samples, 1]).
        :type y_pred: np.ndarray

        :return: Массив вероятностей [p, 1 - p] (размер: [n_samples, 2]).
        :rtype: np.ndarray
        """
        # Если и так уже две вероятности, то менять ничего не надо
        if y_pred.shape[1] == 2:
            return y_pred
        
        # Если размер [n_samples,], то переводим в [n_samples, 1]
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)

        p = y_pred[:, 0]
        return np.column_stack([1 - p, p])

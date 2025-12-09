import numpy as np

from src.losses import ABCLoss


class CrossEntropyLoss(ABCLoss):
    def __init__(self, is_binary: bool = False, type_logits: str | None = None):
        super().__init__()

        self.is_binary = is_binary
        assert (type_logits is None) or (type_logits in ['softmax', 'log_softmax']), f'Параметр type_logits может принимать значения [None, "softmax", "log_softmax"]'
        self.type_logits = type_logits

    def __call__(self, y_true: np.ndarray, logits: np.ndarray) -> float:
        """
        Подсчёт ошибки.
        
        :param y_true: Истинные метки класса. Форма [n_batch,] или [n_batch, 1].
        :type y_true: np.ndarray
        :param logits: Предсказанные значения [0-1] для анализа. Форма [n_batch, n_classes].
        :type logits: np.ndarray
        
        :return: Значение ошибки.
        :rtype: float
        """
        # Проверка размерностей
        assert (y_true.ndim == 1) or ((y_true.ndim == 2) and (y_true.shape[1] == 1)), f'Форма целей должна быть [n_batch,] или [n_batch, 1]'
        if self.is_binary:
            assert (logits.ndim == 2) and (logits.shape[1] <= 2), f'Для бинарной классификации формы логитов [n_batch, 1] или [n_batch, 2]'
        else:
            assert (logits.ndim == 2) and (logits.shape[1] > 2), f'Для многоклассовой классификации формы логитов [n_batch, n_classes]'

        # Берём кол-во объектов и приводим метки к форме [n_batch,]
        n_batch = logits.shape[0]
        y_true = y_true.flatten()
        if self.learning: self.y_true = y_true.copy()

        # Считаем ошибку для объектов
        if self.is_binary and logits.shape[1] == 1:
            # Для бинарной классификации с 1-м выходом у модели
            logits = logits.flatten()
            loss = -y_true*np.log(logits) - (1 - y_true)*np.log((1 - logits))
        else:
            # Для многоклассовой классификации и бинарной с 2-я выходами модели: log_softmax + NLL
            if self.type_logits == 'softmax':
                # Логиты - это распределение вероятностей
                loss = -np.log(logits[np.arange(n_batch), y_true])
            elif self.type_logits == 'log_softmax':
                # Логиты - это логарифмированное распределение вероятностей
                loss = -logits[np.arange(n_batch), y_true]
            else:
                # Логиты - значения с последнего слоя
                shifted = logits - np.max(logits, axis=-1, keepdims=True)
                softmax = np.exp(shifted) / np.sum(np.exp(shifted), axis=-1, keepdims=True)
                log_softmax = np.log(softmax)

                loss = -log_softmax[np.arange(n_batch), y_true]

        return np.mean(loss)

    def backward_pass(self, logits: np.ndarray) -> np.ndarray:
        """
        Частная производная функции потерь по входам.
        
        :param logits: Предсказанные значения. \
            Если self.type_logits == 'softmax', значит это вероятности полученные с помощью применения функции Softmax. \
            Если self.type_logits == 'log_softmax', значит это логарифмированные вероятности полученные с помощью применения функции LogSoftmax. \
            Если self.type_logits == None, значит это обычные предсказания, а не распределения вероятностей.
        :type logits: np.ndarray

        :return: Значения частной производной.
        :rtype: np.ndarray
        """
        n_batch = len(logits)
        
        # Подсчёт частной производной по входам
        if self.is_binary and logits.shape[1] == 1:
            # Для бинарной классификации с 1-м выходом у модели
            logits = logits.flatten()
            dL = -self.y_true/logits + (1 - self.y_true)/(1 - logits)
        else:
            # Для многоклассовой классификации и бинарной с 2-я выходами модели
            if self.type_logits == 'softmax':
                # Логиты - это распределение вероятностей
                dL = logits.copy()
            elif self.type_logits == 'log_softmax':
                # Логиты - это логарифмированное распределение вероятностей
                softmax = np.exp(logits)
                dL = softmax.copy()
            else:
                # Логиты - значения с последнего слоя
                shifted = logits - np.max(logits, axis=-1, keepdims=True)
                softmax = np.exp(shifted) / np.sum(np.exp(shifted), axis=-1, keepdims=True)
                dL = softmax.copy()
            
            # dL = p − one_hot(y)
            dL[np.arange(n_batch), self.y_true] -= 1

        return dL/n_batch

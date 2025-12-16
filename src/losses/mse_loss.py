import numpy as np

from src.losses import ABCLoss


class MSELoss(ABCLoss):
    """
    MSELoss с поддержкой RNN:
    y_true: (n_batch,)
    logits: (n_batch, n_time, n_features) или (n_batch, n_time, 1)
    """

    # def __call__(self, y_true: np.ndarray, logits: np.ndarray) -> float:
    #     if self.learning:
    #         self.logits_shape = logits.shape
    #         # Расширяем y_true на временную ось
    #         if y_true.ndim == 1 and logits.ndim == 3:
    #             self.y_true = np.repeat(y_true[:, None, None], repeats=logits.shape[1], axis=1)
    #             if logits.shape[2] != 1:
    #                 # Если многоклассовая/многовекторная регрессия, повторяем по features
    #                 self.y_true = np.repeat(self.y_true, repeats=logits.shape[2], axis=2)
    #         else:
    #             self.y_true = y_true.copy()
        
    #     return np.mean((self.y_true - logits) ** 2)
    def __call__(self, y_true: np.ndarray, logits: np.ndarray) -> float:
        if logits.ndim == 3:
            # Меняем формы y_true в зависимости от задачи
            if (y_true.ndim == 1) or ((y_true.ndim == 2) and (y_true.shape[1] == 1)):
                # Если задача many-to-one
                pass
            elif (y_true.ndim == 2) and (y_true.shape[1] > 1):
                # Если задача many-to-many
                y_true = y_true[:, :, None]

        if self.learning:
            self.y_true = y_true.copy()
        
        return np.mean((logits - y_true) ** 2)

    def backward_pass(self, logits: np.ndarray) -> np.ndarray:
        n_elements = np.prod(self.y_true.shape)
        return 2 * (logits - self.y_true) / n_elements


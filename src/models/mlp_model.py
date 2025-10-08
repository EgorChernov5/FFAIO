import numpy as np
from collections.abc import Callable

from src.layers import Layer
from src.losses import Loss
from src.optimizers import Optimizer


class MLP:
    def __init__(self, arch_model: list[Layer], loss: Loss, optimizer: Optimizer):
        self.arch_model = arch_model
        self.loss = loss
        self.optimizer = optimizer
        self.optimizer.add_params(self.arch_model, self.loss)

    def len_weights(self):
        return sum([layer.len_weights() for layer in self.arch_model])

    def predict(self, inputs: np.ndarray, postprocess: Callable[[np.ndarray], np.ndarray] | None = None) -> np.ndarray:
        """
        Получение предсказание модели.

        :param inputs: Объекты с признаками.
        :type inputs: np.ndarray

        :return: Предсказания модели.
        :rtype: np.ndarray
        """
        res = inputs.copy()
        for layer in self.arch_model:
            res = layer(res)

        return postprocess(res) if postprocess is not None else res

    def train_model(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: np.ndarray,
            y_val: np.ndarray,
            n_epochs: int,
            postprocess: Callable[[np.ndarray], np.ndarray] | None = None,
            count_metric: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
            verbose_n_batch_multiple: int = 1
        ):
        for n_epoch in range(n_epochs):
            losses = []
            metrics = []
            # Берём mini-batch из тренировочной выборки
            for n_batch, X_train_batch, y_train_batch in self.optimizer.data_loader.get_data(X_train, y_train):
                y_pred_batch = self.predict(X_train_batch)

                loss_batch = self.loss(y_train_batch, y_pred_batch)

                self.optimizer.step()
                self.optimizer.zero_grad()

                if verbose_n_batch_multiple and n_batch%verbose_n_batch_multiple == 0:
                    losses.append(loss_batch)
                    y_pred = self.predict(X_val, postprocess)
                    metrics.append(count_metric(y_val, y_pred))
                    print(f"Epoch {n_epoch + 1} ({n_batch*self.optimizer.data_loader.batch_size}/{len(y_train)}): \
                          {self.loss.to_str()} = {round(np.mean(losses), 3)} \
                          {count_metric.__name__} = {round(np.mean(metrics), 3)}")

from typing import Callable
import numpy as np


def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_epochs: int,
        model,
        loss,
        optimizer,
        postprocess: Callable[[np.ndarray], np.ndarray] | None = None,
        count_metric: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
        verbose_n_batch_multiple: int = 1
    ):
    for n_epoch in range(n_epochs):
        losses = []
        metrics = []
        # Берём mini-batch из тренировочной выборки
        for n_batch, X_train_batch, y_train_batch in enumerate(optimizer.data_loader.get_data(X_train, y_train)):
            y_pred_batch = model.predict(X_train_batch)

            losses.append(loss(y_train_batch, y_pred_batch))

            optimizer.step()

            if verbose_n_batch_multiple and n_batch%verbose_n_batch_multiple == 0:
                y_pred = model.predict(X_val, postprocess)
                metrics.append(count_metric(y_val, y_pred))
                print(f'Epoch {n_epoch}: ({n_batch*self.data_loader.batch_size}/{len(y_train)}): \
                      {self.loss.to_str()} = {round(np.mean(losses), 3)}\
                        {count_metric.__name__} = {round(np.mean(metrics), 3)}')

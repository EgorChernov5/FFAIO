import numpy as np
from abc import ABC
from typing import Callable

from src.layers import ABCLayer
from src.activations import ABCActivation
from src.structure_layers import ABCStructureLayer
from src.cells import ABCCell
from src.losses import ABCLoss
from src.optimizers import ABCOptimizer
from src.regularizers import ABCRegularizer


class ABCModel(ABC):
    def __init__(self, arch_model: list[ABCLayer| ABCActivation | ABCStructureLayer | ABCCell]):
        self.arch_model = arch_model

        self.learning = True
        self.losses_train = []
        self.losses_val = []
        self.metrics_val = []

    def __call__(self, X: np.ndarray, postprocess: Callable[[np.ndarray], np.ndarray] | None = None) -> np.ndarray:
        outputs = X.copy()
        prev_hidden_state = None
        for layer in self.arch_model:
            if isinstance(layer, ABCCell):
                outputs, prev_hidden_state = layer(outputs, prev_hidden_state)
            else:
                outputs = layer(outputs)
                prev_hidden_state = None

        return outputs if postprocess is None else postprocess(outputs)
    
    def _count_verbose(
            self,
            X_train, y_train, X_val, y_val,
            n_epoch, n_batch,
            losses_train_batch, losses_val_batch, metrics_val_batch, loss_batch,
            optimizer, loss, regularizer, postprocess, count_metric,
            verbose_n_batch_multiple, verbose_statistic
        ):
        # Отключаем сохранение параметров
        self.eval()
        loss.eval()
        if verbose_n_batch_multiple and n_batch%verbose_n_batch_multiple == 0:
            if X_val is not None:
                # Val loss
                y_val_probs = self.__call__(X_val)
                loss_val = loss(y_val, y_val_probs)
                if regularizer: loss_val += regularizer(self.get_weights())
                # Metric
                y_val_preds = y_val_probs if postprocess is None else postprocess(y_val_probs)
                metrics_val_batch.append(count_metric(y_val, y_val_preds))

            if verbose_statistic == 'all':  # Полное сохранение
                losses_train_batch.append(loss_batch)
                if X_val is not None: losses_val_batch.append(loss_val)
            else:
                if len(losses_train_batch) == 0:
                    _, X_rand, y_rand = next(optimizer.data_loader.get_data(X_train, y_train))
                    y_rand_pred = self.__call__(X_rand)
                    loss_rand = loss(y_rand, y_rand_pred)
                    loss_rand = regularizer(self.get_weights(), loss_rand) if regularizer else loss_rand
                    losses_train_batch = [loss_rand]
                    if X_val is not None: losses_val_batch = [loss_rand]
                
                lambda_q = 0.1
                if verbose_statistic == 'SMA':
                    m = 10
                    lambda_q = 1/m
                elif verbose_statistic == 'EMA':
                    m = 10
                    lambda_q = 2/(m + 1)

                # Train loss
                losses_train_batch = [lambda_q*loss_batch + (1 - lambda_q)*losses_train_batch[0]]
                # Val loss
                if X_val is not None: losses_val_batch = [lambda_q*loss_val + (1 - lambda_q)*losses_val_batch[0]]

            # Visualize
            if X_val is not None:
                print(f"Epoch {n_epoch + 1} ({n_batch*optimizer.data_loader.batch_size}/{len(y_train)}):\t"
                        f"Train {loss.to_str()} = {round(np.mean(losses_train_batch), 3)}\t"
                        f"Val {loss.to_str()} = {round(np.mean(losses_train_batch), 3)}\t"
                        f"{count_metric.__name__} = {round(np.mean(metrics_val_batch), 3)}")
            else:
                print(f"Epoch {n_epoch + 1} ({n_batch*optimizer.data_loader.batch_size}/{len(y_train)}):\t"
                        f"Train {loss.to_str()} = {round(np.mean(losses_train_batch), 3)}")
        
        # Включаем сохранение параметров
        self.train()
        loss.train()
        return losses_train_batch, losses_val_batch, metrics_val_batch
    
    def get_weights_layers(self) -> list[ABCLayer]:
        return [struc_element for struc_element in self.arch_model if isinstance(struc_element, (ABCLayer, ABCCell))]
    
    def get_weights(self) -> dict:
        weights = {}
        i = 1
        prev_layer = None
        for layer in self.get_weights_layers():
            name_layer = layer.to_str()
            i = 1 if (prev_layer is None) or (name_layer not in prev_layer) else i + 1
            prev_layer = name_layer

            if isinstance(layer, ABCLayer):
                W, b = layer.get_weights()

                name_layer = name_layer + str(i)
                weights[f'{name_layer}.weight'] = W
                if b is not None: weights[f'{name_layer}.bias'] = b
            else:
                W, b, Wh, bh = layer.get_weights()

                weights[f'{name_layer}.weight_ih_l{i}'] = W
                weights[f'{name_layer}.weight_hh_l{i}'] = Wh
                if layer.bias:
                    weights[f'{name_layer}.bias_ih_l{i}'] = b
                    weights[f'{name_layer}.bias_hh_l{i}'] = bh

        return weights
    
    def backward_pass(self, loss: ABCLoss):
        delta = loss.backward_pass(self.arch_model[-1].outputs)
        # for layer in reversed(self.arch_model[:-1]):
        for layer in reversed(self.arch_model):
            # Считаем дельта правило или градиент весов и передаём ошибку дальше влево
            delta = layer.backward_pass(delta)

    def train_model(
            self,
            n_epochs: int,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: np.ndarray | None,
            y_val: np.ndarray | None,
            loss: ABCLoss,
            optimizer: ABCOptimizer,
            regularizer: ABCRegularizer | None = None,
            postprocess: Callable[[np.ndarray], np.ndarray] | None = None,
            count_metric: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
            verbose_n_batch_multiple: int = 1,
            verbose_statistic: str = 'all'
        ):
        self.train()
        loss.train()
        for n_epoch in range(n_epochs):
            losses_train_batch = []
            losses_val_batch = []
            metrics_val_batch = []
            for n_batch, X_train_batch, y_train_batch in optimizer.data_loader.get_data(X_train, y_train):
                # forward pass
                y_pred_batch = self.__call__(X_train_batch)
                loss_batch = loss(y_train_batch, y_pred_batch)
                if regularizer: loss_batch += regularizer(self.get_weights())

                # backward pass
                self.backward_pass(loss)
                optimizer.step(regularizer)

                # verbose
                losses_train_batch, losses_val_batch, metrics_val_batch = self._count_verbose(
                    X_train, y_train, X_val, y_val,
                    n_epoch, n_batch,
                    losses_train_batch, losses_val_batch, metrics_val_batch, loss_batch,
                    optimizer, loss, regularizer, postprocess, count_metric,
                    verbose_n_batch_multiple, verbose_statistic
                )
                    
            self.losses_train.append(np.mean(losses_train_batch))
            if X_val is not None:
                self.losses_val.append(np.mean(losses_val_batch))
                self.metrics_val.append(np.mean(metrics_val_batch))

    def train(self):
        self.learning = True
        for layer in self.arch_model:
            if hasattr(layer, "train") and callable(layer.train):
                layer.train()

    def eval(self):
        self.learning = False
        for layer in self.arch_model:
            if hasattr(layer, "eval") and callable(layer.eval):
                layer.eval()

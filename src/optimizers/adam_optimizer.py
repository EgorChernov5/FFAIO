import numpy as np

from src.optimizers import ABCOptimizer
from src.regularizers import ABCRegularizer
from src.layers import ABCLayer
from src.cells import ABCCell
from src.data_loaders import ABCLoader


class AdamOptimizer(ABCOptimizer):
    def __init__(
        self,
        model_weights_layers: list[ABCLayer | ABCCell],
        data_loader: ABCLoader | None = None,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8
    ):
        super().__init__(model_weights_layers, data_loader, lr)
        
        self.beta1 = beta1  # для экспоненциального скользящего среднего градиента
        self.beta2 = beta2  # для экспоненциального скользящего среднего квадрата градиента
        self.epsilon = epsilon

        self.m_buffers = [None] * len(self.model_weights_layers)
        self.v_buffers = [None] * len(self.model_weights_layers)
        self.t = 0  # номер шага

    # def step(self, regularizer: ABCRegularizer | None = None):
    #     self.t += 1
    #     for i, layer in enumerate(self.model_weights_layers):
    #         # Получаем веса и градиенты
    #         if isinstance(layer, ABCLayer):
    #             W, b = layer.get_weights()
    #             gradW, gradb = layer.get_gradients()
    #         elif isinstance(layer, ABCCell):
    #             W, b, Wh, bh = layer.get_weights()
    #             gradW, gradb, gradWh, gradbh = layer.get_gradients()

    #         # Применение регуляризатора
    #         if regularizer:
    #             if isinstance(layer, ABCLayer):
    #                 reg_W, reg_b = regularizer.pd_wrt_w((W, b))
    #                 gradW += reg_W
    #                 if b is not None: gradb += reg_b
    #             elif isinstance(layer, ABCCell):
    #                 reg_W, reg_b = regularizer.pd_wrt_w((W, b))
    #                 reg_Wh, reg_bh = regularizer.pd_wrt_w((Wh, bh))
    #                 gradW += reg_W
    #                 gradWh += reg_Wh
    #                 if b is not None: gradb += reg_b
    #                 if bh is not None: gradbh += reg_bh

    #         # Инициализация буферов m и v
    #         if self.m_buffers[i] is None:
    #             if isinstance(layer, ABCLayer):
    #                 self.m_buffers[i] = (np.zeros_like(W), np.zeros_like(b) if b is not None else None)
    #                 self.v_buffers[i] = (np.zeros_like(W), np.zeros_like(b) if b is not None else None)
    #             elif isinstance(layer, ABCCell):
    #                 self.m_buffers[i] = (
    #                     np.zeros_like(W), np.zeros_like(b) if b is not None else None,
    #                     np.zeros_like(Wh), np.zeros_like(bh) if bh is not None else None
    #                 )
    #                 self.v_buffers[i] = (
    #                     np.zeros_like(W), np.zeros_like(b) if b is not None else None,
    #                     np.zeros_like(Wh), np.zeros_like(bh) if bh is not None else None
    #                 )

    #         # Обновление моментов и корректировка весов
    #         if isinstance(layer, ABCLayer):
    #             mW, mb = self.m_buffers[i]
    #             vW, vb = self.v_buffers[i]

    #             mW = self.beta1 * mW + (1 - self.beta1) * gradW
    #             vW = self.beta2 * vW + (1 - self.beta2) * (gradW ** 2)
    #             if b is not None:
    #                 mb = self.beta1 * mb + (1 - self.beta1) * gradb
    #                 vb = self.beta2 * vb + (1 - self.beta2) * (gradb ** 2)

    #             # Сохраняем буферы
    #             self.m_buffers[i] = (mW, mb)
    #             self.v_buffers[i] = (vW, vb)

    #             # Обновление весов
    #             W -= self.lr * (mW / (1 - self.beta1 ** self.t)) / (np.sqrt(vW / (1 - self.beta2 ** self.t)) + self.epsilon)
    #             if b is not None:
    #                 b -= self.lr * (mb / (1 - self.beta1 ** self.t)) / (np.sqrt(vb / (1 - self.beta2 ** self.t)) + self.epsilon)

    #             layer.update_weights(W, b)

    #         elif isinstance(layer, ABCCell):
    #             mW, mb, mWh, mbh = self.m_buffers[i]
    #             vW, vb, vWh, vbh = self.v_buffers[i]

    #             # моменты
    #             mW = self.beta1 * mW + (1 - self.beta1) * gradW
    #             vW = self.beta2 * vW + (1 - self.beta2) * (gradW ** 2)
    #             mWh = self.beta1 * mWh + (1 - self.beta1) * gradWh
    #             vWh = self.beta2 * vWh + (1 - self.beta2) * (gradWh ** 2)
    #             if b is not None:
    #                 mb = self.beta1 * mb + (1 - self.beta1) * gradb
    #                 vb = self.beta2 * vb + (1 - self.beta2) * (gradb ** 2)
    #             if bh is not None:
    #                 mbh = self.beta1 * mbh + (1 - self.beta1) * gradbh
    #                 vbh = self.beta2 * vbh + (1 - self.beta2) * (gradbh ** 2)

    #             self.m_buffers[i] = (mW, mb, mWh, mbh)
    #             self.v_buffers[i] = (vW, vb, vWh, vbh)

    #             # обновление весов
    #             W -= self.lr * (mW / (1 - self.beta1 ** self.t)) / (np.sqrt(vW / (1 - self.beta2 ** self.t)) + self.epsilon)
    #             Wh -= self.lr * (mWh / (1 - self.beta1 ** self.t)) / (np.sqrt(vWh / (1 - self.beta2 ** self.t)) + self.epsilon)
    #             if b is not None:
    #                 b -= self.lr * (mb / (1 - self.beta1 ** self.t)) / (np.sqrt(vb / (1 - self.beta2 ** self.t)) + self.epsilon)
    #             if bh is not None:
    #                 bh -= self.lr * (mbh / (1 - self.beta1 ** self.t)) / (np.sqrt(vbh / (1 - self.beta2 ** self.t)) + self.epsilon)

    #             layer.update_weights(W, b, Wh, bh)
    def step(self, regularizer: ABCRegularizer | None = None):
        self.t += 1

        for i, layer in enumerate(self.model_weights_layers):
            weights = layer.get_weights()
            grads = layer.get_gradients()

            # --- regularizer ---
            if regularizer:
                new_grads = []
                for W, g in zip(weights, grads):
                    if W is not None:
                        reg_W = regularizer.pd_wrt_w((W, None))[0]
                        new_grads.append(g + reg_W)
                    else:
                        new_grads.append(None)
                grads = tuple(new_grads)

            # --- init buffers ---
            if self.m_buffers[i] is None:
                self.m_buffers[i] = tuple(
                    np.zeros_like(w) if w is not None else None for w in weights
                )
                self.v_buffers[i] = tuple(
                    np.zeros_like(w) if w is not None else None for w in weights
                )

            new_weights = []
            new_m = []
            new_v = []

            for w, g, m, v in zip(weights, grads, self.m_buffers[i], self.v_buffers[i]):
                if w is None:
                    new_weights.append(None)
                    new_m.append(None)
                    new_v.append(None)
                    continue

                m = self.beta1 * m + (1 - self.beta1) * g
                v = self.beta2 * v + (1 - self.beta2) * (g ** 2)

                m_hat = m / (1 - self.beta1 ** self.t)
                v_hat = v / (1 - self.beta2 ** self.t)

                w = w - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

                new_weights.append(w)
                new_m.append(m)
                new_v.append(v)

            self.m_buffers[i] = tuple(new_m)
            self.v_buffers[i] = tuple(new_v)

            layer.update_weights(*new_weights)


import numpy as np

from src.optimizers import ABCOptimizer
from src.regularizers import ABCRegularizer
from src.data_loaders import ABCLoader
from src.layers import ABCLayer


class AdamOptimizer(ABCOptimizer):
    def __init__(
        self,
        model_weights_layers: list[ABCLayer],
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

        # Для каждого слоя создаём буферы m и v
        self.m_buffers = [None] * len(self.model_weights_layers)
        self.v_buffers = [None] * len(self.model_weights_layers)
        self.t = 0  # номер шага

    def step(self, regularizer: ABCRegularizer | None = None):
        self.t += 1
        for i, layer in enumerate(self.model_weights_layers):
            W, b = layer.get_weights()
            gradW, gradb = layer.get_gradients()

            # Применение регуляризатора
            if regularizer is not None:
                reg_W, reg_b = regularizer.pd_wrt_w((W, b))
                gradW = gradW + reg_W
                if b is not None:
                    gradb = gradb + reg_b

            # Инициализация буферов m и v
            if self.m_buffers[i] is None:
                mW = np.zeros_like(W)
                vW = np.zeros_like(W)
                mb = np.zeros_like(b) if b is not None else None
                vb = np.zeros_like(b) if b is not None else None
                self.m_buffers[i] = (mW, mb)
                self.v_buffers[i] = (vW, vb)

            mW, mb = self.m_buffers[i]
            vW, vb = self.v_buffers[i]

            # Обновление моментов
            mW = self.beta1 * mW + (1 - self.beta1) * gradW
            vW = self.beta2 * vW + (1 - self.beta2) * (gradW ** 2)
            if b is not None:
                mb = self.beta1 * mb + (1 - self.beta1) * gradb
                vb = self.beta2 * vb + (1 - self.beta2) * (gradb ** 2)

            # Сохранение буферов
            self.m_buffers[i] = (mW, mb)
            self.v_buffers[i] = (vW, vb)

            # Коррекция смещения
            mW_hat = mW / (1 - self.beta1 ** self.t)
            vW_hat = vW / (1 - self.beta2 ** self.t)
            W -= self.lr * mW_hat / (np.sqrt(vW_hat) + self.epsilon)

            if b is not None:
                mb_hat = mb / (1 - self.beta1 ** self.t)
                vb_hat = vb / (1 - self.beta2 ** self.t)
                b -= self.lr * mb_hat / (np.sqrt(vb_hat) + self.epsilon)

            layer.update_weights(W, b)

import numpy as np

from src.optimizers import ABCOptimizer
from src.regularizers import ABCRegularizer
from src.data_loaders import ABCLoader
from src.layers import ABCLayer
from src.cells import ABCCell


class GDOptimizer(ABCOptimizer):
    def __init__(
            self,
            model_weights_layers: list[ABCLayer | ABCCell],
            data_loader: ABCLoader | None = None,
            lr: float = 0.001,
            momentum: float = 0.,
            dampening: float = 0.
        ):
        super().__init__(model_weights_layers, data_loader, lr)
        self.momentum = momentum
        self.dampening = dampening
        self.momentum_buffers = [None] * len(self.model_weights_layers)

    def step(self, regularizer: ABCRegularizer | None = None):
        for i, layer in enumerate(self.model_weights_layers):
            # Получаем веса и градиенты
            if isinstance(layer, ABCLayer):
                W, b = layer.get_weights()
                gradW, gradb = layer.get_gradients()
            elif isinstance(layer, ABCCell):
                W, b, Wh, bh = layer.get_weights()
                gradW, gradb, gradWh, gradbh = layer.get_gradients()

            # Применяем регуляризатор
            if regularizer:
                if isinstance(layer, ABCLayer):
                    reg_W, reg_b = regularizer.pd_wrt_w((W, b))
                    gradW += reg_W
                    if b is not None: gradb += reg_b
                elif isinstance(layer, ABCCell):
                    reg_W, reg_b = regularizer.pd_wrt_w((W, b))
                    reg_Wh, reg_bh = regularizer.pd_wrt_w((Wh, bh))
                    gradW += reg_W
                    gradWh += reg_Wh
                    if b is not None: gradb += reg_b
                    if bh is not None: gradbh += reg_bh

            # Применение momentum
            if self.momentum:
                if self.momentum_buffers[i] is None:
                    if isinstance(layer, ABCLayer):
                        vW = np.zeros_like(W)
                        vb = np.zeros_like(b) if b is not None else None
                        self.momentum_buffers[i] = (vW, vb)
                    elif isinstance(layer, ABCCell):
                        vW = np.zeros_like(W)
                        vb = np.zeros_like(b) if b is not None else None
                        vWh = np.zeros_like(Wh)
                        vbh = np.zeros_like(bh) if bh is not None else None
                        self.momentum_buffers[i] = (vW, vb, vWh, vbh)

                # Обновляем буферы
                if isinstance(layer, ABCLayer):
                    vW, vb = self.momentum_buffers[i]
                    vW = self.momentum*vW + (1 - self.dampening)*gradW
                    if b is not None:
                        vb = self.momentum*vb + (1 - self.dampening)*gradb
                    self.momentum_buffers[i] = (vW, vb)
                    gradW, gradb = vW, vb
                elif isinstance(layer, ABCCell):
                    vW, vb, vWh, vbh = self.momentum_buffers[i]
                    vW = self.momentum*vW + (1 - self.dampening)*gradW
                    vWh = self.momentum*vWh + (1 - self.dampening)*gradWh
                    if b is not None: vb = self.momentum*vb + (1 - self.dampening)*gradb
                    if bh is not None: vbh = self.momentum*vbh + (1 - self.dampening)*gradbh
                    self.momentum_buffers[i] = (vW, vb, vWh, vbh)
                    gradW, gradb, gradWh, gradbh = vW, vb, vWh, vbh

            # Шаг оптимизации
            if isinstance(layer, ABCLayer):
                W -= self.lr * gradW
                if b is not None: b -= self.lr * gradb
                layer.update_weights(W, b)
            elif isinstance(layer, ABCCell):
                W -= self.lr * gradW
                Wh -= self.lr * gradWh
                if b is not None: b -= self.lr * gradb
                if bh is not None: bh -= self.lr * gradbh
                layer.update_weights(W, b, Wh, bh)

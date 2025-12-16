import numpy as np

from src.cells import ABCCell
from src.activations import ABCActivation, TanhActivation
from src.weights_initializers import random_numbers_init


class RNNCell(ABCCell):
    def __init__(self, in_features, hidden_size, bias: bool = True, activation: ABCActivation | None = None):
        super().__init__()

        self.in_features = in_features
        self.out_features = hidden_size
        self.bias = bias

        # Веса, преобразующие входные данные
        weights = np.array([random_numbers_init(in_features + 1 if bias else in_features) for _ in range(hidden_size)], dtype=np.float32)
        self.W = weights[:, :-1] if bias else weights
        self.b = weights[:, -1] if bias else None
        # Веса, преобразующие прошлое состояние
        weights = np.array([random_numbers_init(hidden_size + 1 if bias else hidden_size) for _ in range(hidden_size)], dtype=np.float32)
        self.Wh = weights[:, :-1] if bias else weights
        self.bh = weights[:, -1] if bias else None
        # Активация
        self.activation = TanhActivation() if activation is None else activation

        # Градиенты
        self.gradWh: np.ndarray | None = None
        self.gradbh: np.ndarray | None = None

        # Для BPTT
        self.x_list = []
        self.z_list = []
        self.h_list = []
        self.prev_hidden_state = None  # скрытое состояние для начала последовательности (n_batch, out_features)

    def __call__(self, inputs: np.ndarray, prev_hidden_state: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """
        inputs: (n_batch, n_time, in_features)
        prev_hidden_state: (n_batch, out_features) или None
        returns: outputs (n_batch, n_time, out_features), last_hidden_state (n_batch, out_features)
        """
        # сброс tanh стора
        self.activation.reset()
        n_batch, n_time, _ = inputs.shape

        # Инициализация скрытого состояния
        hidden_state = np.zeros((n_batch, self.out_features), dtype=np.float32) if prev_hidden_state is None else prev_hidden_state

        if self.learning:
            self.inputs = inputs.copy()
            self.x_list = []
            self.h_list = []
            self.z_list = []
            self.prev_hidden_state = hidden_state.copy()

        outputs = []
        for t in range(n_time):
            input_t = inputs[:, t, :]
            Z_t = input_t@self.W.T + hidden_state@self.Wh.T
            if self.bias:
                Z_t += self.b + self.bh

            hidden_state = self.activation(Z_t)

            if self.learning:
                self.x_list.append(input_t)
                self.z_list.append(Z_t)
                self.h_list.append(hidden_state)

            outputs.append(hidden_state)

        outputs = np.stack(outputs, axis=1)

        if self.learning:
            self.outputs = outputs.copy()

        return outputs, hidden_state

    def get_weights(self) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray | None]:
         return self.W, self.b, self.Wh, self.bh

    def get_gradients(self) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray | None]:
        return self.gradW, self.gradb, self.gradWh, self.gradbh
    
    def update_weights(self, W: np.ndarray, b: np.ndarray | None, Wh: np.ndarray, bh: np.ndarray | None):
        self.W, self.b = W, b
        self.Wh, self.bh = Wh, bh
    
    def backward_pass(self, delta: np.ndarray) -> np.ndarray:
        n_batch, n_time, _ = delta.shape

        grad_x = np.zeros((n_batch, n_time, self.in_features), dtype=np.float32)
        gradW = np.zeros_like(self.W, dtype=np.float32)
        gradWh = np.zeros_like(self.Wh, dtype=np.float32)
        gradb = np.zeros_like(self.b, dtype=np.float32) if self.bias else None
        gradbh = np.zeros_like(self.bh, dtype=np.float32) if self.bias else None

        delta_h_next = np.zeros((n_batch, self.out_features), dtype=np.float32)
        for t in reversed(range(n_time)):
            # dL/dh_t = delta_from_loss + delta_from_next_time
            delta_h = delta[:, t, :] + delta_h_next
            # получаем dZ через activation
            dZ = self.activation.backward_pass(delta_h, t)

            # dL/dx_t = dZ @ W
            grad_x[:, t, :] = dZ @ self.W

            # предыдущий скрытый (после активации)
            h_prev = self.prev_hidden_state if t == 0 else self.h_list[t-1]

            # аккумулируем градиенты по параметрам (сумма по батч и time)
            gradW += dZ.T @ self.x_list[t]
            gradWh += dZ.T @ h_prev

            if self.bias:
                gradb += dZ.sum(axis=0)
                gradbh += dZ.sum(axis=0)

            # propagate to previous hidden
            delta_h_next = dZ @ self.Wh

        self.gradW = gradW
        self.gradWh = gradWh
        self.gradb = gradb
        self.gradbh = gradbh

        return grad_x.astype(np.float32)

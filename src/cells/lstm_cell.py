import numpy as np

from src.cells import ABCCell
from src.activations import SigmoidActivation, TanhActivation
from src.weights_initializers import random_numbers_init


class LSTMCell(ABCCell):
    def __init__(self, in_features, hidden_size, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = hidden_size
        self.bias = bias

        # ===== Weights =====
        def init_gate(in_f, out_f):
            W = np.array([random_numbers_init(in_f) for _ in range(out_f)], dtype=np.float32)
            Wh = np.array([random_numbers_init(out_f) for _ in range(out_f)], dtype=np.float32)
            b = np.zeros(out_f, dtype=np.float32) if bias else None
            bh = np.zeros(out_f, dtype=np.float32) if bias else None
            return W, b, Wh, bh

        self.Wi, self.bi, self.Whi, self.bhi = init_gate(in_features, hidden_size)
        self.Wf, self.bf, self.Whf, self.bhf = init_gate(in_features, hidden_size)
        self.Wo, self.bo, self.Who, self.bho = init_gate(in_features, hidden_size)
        self.Wg, self.bg, self.Whg, self.bhg = init_gate(in_features, hidden_size)

        # ===== Activations =====
        self.sigmoid = SigmoidActivation()
        self.tanh = TanhActivation()

        # ===== BPTT storage =====
        self.x_list = []
        self.i_list = []
        self.f_list = []
        self.o_list = []
        self.g_list = []
        self.c_list = []
        self.h_list = []

        self.h_prev = None
        self.c_prev = None

    # -------------------------------------------------
    # Forward
    # -------------------------------------------------
    def __call__(self, inputs: np.ndarray,
                 prev_hidden_state: tuple[np.ndarray, np.ndarray] | None = None):
        """
        inputs: (batch, time, in_features)
        prev_hidden_state: (h0, c0)
        """
        self.sigmoid.reset()
        self.tanh.reset()

        n_batch, n_time, _ = inputs.shape

        if prev_hidden_state is None:
            h_t = np.zeros((n_batch, self.out_features), dtype=np.float32)
            c_t = np.zeros_like(h_t)
        else:
            h_t, c_t = prev_hidden_state

        if self.learning:
            self.x_list, self.i_list, self.f_list = [], [], []
            self.o_list, self.g_list, self.c_list, self.h_list = [], [], [], []
            self.h_prev = h_t.copy()
            self.c_prev = c_t.copy()

        outputs = []

        for t in range(n_time):
            x_t = inputs[:, t, :]

            # forget gate
            f_t = self.sigmoid(x_t@self.Wf.T + h_t@self.Whf.T + self.bf + self.bhf)
            # input gate (sigmoid определяет важность признаков, а tanh их значение)
            i_t = self.sigmoid(x_t@self.Wi.T + h_t@self.Whi.T + self.bi + self.bhi)
            g_t = self.tanh(x_t@self.Wg.T + h_t@self.Whg.T + self.bg + self.bhg)
            # output gate
            o_t = self.sigmoid(x_t@self.Wo.T + h_t@self.Who.T + self.bo + self.bho)

            c_t = f_t*c_t + i_t*g_t
            h_t = o_t*np.tanh(c_t)

            if self.learning:
                self.x_list.append(x_t)
                self.i_list.append(i_t)
                self.f_list.append(f_t)
                self.o_list.append(o_t)
                self.g_list.append(g_t)
                self.c_list.append(c_t)
                self.h_list.append(h_t)

            outputs.append(h_t)

        outputs = np.stack(outputs, axis=1)
        if self.learning:
            self.outputs = outputs.copy()

        return outputs, (h_t, c_t)

    # -------------------------------------------------
    # Weights API
    # -------------------------------------------------
    def get_weights(self):
        return (
            self.Wi, self.bi, self.Whi, self.bhi,
            self.Wf, self.bf, self.Whf, self.bhf,
            self.Wo, self.bo, self.Who, self.bho,
            self.Wg, self.bg, self.Whg, self.bhg
        )

    def get_gradients(self):
        return self.grads

    def update_weights(self, *weights):
        (
            self.Wi, self.bi, self.Whi, self.bhi,
            self.Wf, self.bf, self.Whf, self.bhf,
            self.Wo, self.bo, self.Who, self.bho,
            self.Wg, self.bg, self.Whg, self.bhg
        ) = weights

    # -------------------------------------------------
    # Backward (BPTT)
    # -------------------------------------------------
    def backward_pass(self, delta: np.ndarray) -> np.ndarray:
        n_batch, n_time, _ = delta.shape

        grad_x = np.zeros((n_batch, n_time, self.in_features), dtype=np.float32)

        grads = [np.zeros_like(w) if w is not None else None for w in self.get_weights()]

        dh_next = np.zeros((n_batch, self.out_features), dtype=np.float32)
        dc_next = np.zeros_like(dh_next)

        for t in reversed(range(n_time)):
            dh = delta[:, t, :] + dh_next
            c_t = self.c_list[t]
            c_prev = self.c_prev if t == 0 else self.c_list[t - 1]

            i_t, f_t = self.i_list[t], self.f_list[t]
            o_t, g_t = self.o_list[t], self.g_list[t]

            tanh_c = np.tanh(c_t)

            do = dh * tanh_c
            dc = dh * o_t * (1 - tanh_c**2) + dc_next

            df = dc * c_prev
            di = dc * g_t
            dg = dc * i_t

            do_raw = self.sigmoid.backward_pass(do, t)
            df_raw = self.sigmoid.backward_pass(df, t)
            di_raw = self.sigmoid.backward_pass(di, t)
            dg_raw = self.tanh.backward_pass(dg, t)

            x_t = self.x_list[t]
            h_prev = self.h_prev if t == 0 else self.h_list[t - 1]

            # ---- grads ----
            for idx, (d, W) in enumerate([
                (di_raw, self.Wi), (di_raw, self.Whi),
                (df_raw, self.Wf), (df_raw, self.Whf),
                (do_raw, self.Wo), (do_raw, self.Who),
                (dg_raw, self.Wg), (dg_raw, self.Whg),
            ]):
                grads[2 * idx] += d.T @ x_t if W.shape[1] == self.in_features else d.T @ h_prev
                if self.bias:
                    grads[2 * idx + 1] += d.sum(axis=0)

            # ---- propagate ----
            dh_next = (
                di_raw @ self.Whi +
                df_raw @ self.Whf +
                do_raw @ self.Who +
                dg_raw @ self.Whg
            )

            dc_next = dc * f_t

            grad_x[:, t, :] = (
                di_raw @ self.Wi +
                df_raw @ self.Wf +
                do_raw @ self.Wo +
                dg_raw @ self.Wg
            )

        self.grads = tuple(grads)
        return grad_x

import numpy as np

from src.cells import ABCCell
from src.activations import ABCActivation, SigmoidActivation, TanhActivation
from src.weights_initializers import random_numbers_init


class GRUCell(ABCCell):
    def __init__(self, in_features, hidden_size, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = hidden_size
        self.bias = bias

        # Update gate weights
        self.Wz = np.array([random_numbers_init(in_features) for _ in range(hidden_size)], dtype=np.float32)
        self.Whz = np.array([random_numbers_init(hidden_size) for _ in range(hidden_size)], dtype=np.float32)
        self.bz = np.zeros(hidden_size, dtype=np.float32) if bias else None
        self.bhz = np.zeros(hidden_size, dtype=np.float32) if bias else None

        # Reset gate weights
        self.Wr = np.array([random_numbers_init(in_features) for _ in range(hidden_size)], dtype=np.float32)
        self.Whr = np.array([random_numbers_init(hidden_size) for _ in range(hidden_size)], dtype=np.float32)
        self.br = np.zeros(hidden_size, dtype=np.float32) if bias else None
        self.bhr = np.zeros(hidden_size, dtype=np.float32) if bias else None

        # Candidate hidden weights
        self.Wc = np.array([random_numbers_init(in_features) for _ in range(hidden_size)], dtype=np.float32)
        self.Whc = np.array([random_numbers_init(hidden_size) for _ in range(hidden_size)], dtype=np.float32)
        self.bc = np.zeros(hidden_size, dtype=np.float32) if bias else None
        self.bhc = np.zeros(hidden_size, dtype=np.float32) if bias else None

        # Activations
        self.sigmoid = SigmoidActivation()
        self.tanh = TanhActivation()

        # BPTT storage
        self.x_list, self.z_list, self.r_list, self.h_tilde_list, self.h_list = [], [], [], [], []
        self.prev_hidden_state = None

    def __call__(self, inputs: np.ndarray, prev_hidden_state: np.ndarray | None = None):
        self.sigmoid.reset()
        self.tanh.reset()
        n_batch, n_time, _ = inputs.shape
        hidden_state = np.zeros((n_batch, self.out_features), dtype=np.float32) if prev_hidden_state is None else prev_hidden_state

        if self.learning:
            self.x_list, self.z_list, self.r_list, self.h_tilde_list, self.h_list = [], [], [], [], []
            self.prev_hidden_state = hidden_state.copy()

        outputs = []
        for t in range(n_time):
            x_t = inputs[:, t, :]

            # Update gate
            z_t = x_t@self.Wz.T + hidden_state@self.Whz.T
            if self.bias: z_t += self.bz + self.bhz
            z_t = self.sigmoid(z_t)

            # Reset gate
            r_t = x_t@self.Wr.T + hidden_state@self.Whr.T
            if self.bias: r_t += self.br + self.bhr
            r_t = self.sigmoid(r_t)

            # Candidate hidden
            h_tilde = x_t@self.Wc.T + (r_t*hidden_state)@self.Whc.T
            if self.bias: h_tilde += self.bc + self.bhc
            h_tilde = self.tanh(h_tilde)

            # Hidden state
            hidden_state = (1 - z_t)*h_tilde + z_t*hidden_state

            if self.learning:
                self.x_list.append(x_t)
                self.z_list.append(z_t)
                self.r_list.append(r_t)
                self.h_tilde_list.append(h_tilde)
                self.h_list.append(hidden_state)

            outputs.append(hidden_state)

        outputs = np.stack(outputs, axis=1)
        if self.learning:
            self.outputs = outputs.copy()
        return outputs, hidden_state

    def get_weights(self):
        return (self.Wz, self.bz, self.Whz, self.bhz,
                self.Wr, self.br, self.Whr, self.bhr,
                self.Wc, self.bc, self.Whc, self.bhc)

    def get_gradients(self):
        return (self.gradWz, self.gradbz, self.gradWhz, self.gradBhz,
                self.gradWr, self.gradbr, self.gradWhr, self.gradBhr,
                self.gradWc, self.gradbc, self.gradWhc, self.gradBhc)

    def update_weights(self, Wz, bz, Whz, bhz, Wr, br, Whr, bhr, Wc, bc, Whc, bhc):
        (self.Wz, self.bz, self.Whz, self.bhz,
         self.Wr, self.br, self.Whr, self.bhr,
         self.Wc, self.bc, self.Whc, self.bhc) = Wz, bz, Whz, bhz, Wr, br, Whr, bhr, Wc, bc, Whc, bhc

    def backward_pass(self, delta: np.ndarray) -> np.ndarray:
        n_batch, n_time, _ = delta.shape

        grad_x = np.zeros((n_batch, n_time, self.in_features), dtype=np.float32)
        gradWz = np.zeros_like(self.Wz)
        gradWhz = np.zeros_like(self.Whz)
        gradbz = np.zeros_like(self.bz) if self.bias else None
        gradbhz = np.zeros_like(self.bhz) if self.bias else None

        gradWr = np.zeros_like(self.Wr)
        gradWhr = np.zeros_like(self.Whr)
        gradbr = np.zeros_like(self.br) if self.bias else None
        gradbhr = np.zeros_like(self.bhr) if self.bias else None

        gradWc = np.zeros_like(self.Wc)
        gradWhc = np.zeros_like(self.Whc)
        gradbc = np.zeros_like(self.bc) if self.bias else None
        gradbhc = np.zeros_like(self.bhc) if self.bias else None

        delta_h_next = np.zeros((n_batch, self.out_features), dtype=np.float32)

        for t in reversed(range(n_time)):
            delta_h = delta[:, t, :] + delta_h_next
            h_prev = self.prev_hidden_state if t == 0 else self.h_list[t-1]

            z_t = self.z_list[t]
            r_t = self.r_list[t]
            h_tilde = self.h_tilde_list[t]

            # dL/dh_tilde
            dh_tilde = delta_h * (1 - z_t)
            dh_tilde_raw = self.tanh.backward_pass(dh_tilde, t)

            # dL/dz
            dz = delta_h * (h_prev - h_tilde)
            dz_raw = self.sigmoid.backward_pass(dz, t)

            # dL/dr
            dr = dh_tilde_raw @ self.Whc * h_prev
            dr_raw = self.sigmoid.backward_pass(dr, t)

            # Gradients accumulation
            gradWc += dh_tilde_raw.T @ self.x_list[t]
            gradWhc += dh_tilde_raw.T @ (r_t * h_prev)
            if self.bias: gradbc += dh_tilde_raw.sum(axis=0); gradbhc += dh_tilde_raw.sum(axis=0)

            gradWz += dz_raw.T @ self.x_list[t]
            gradWhz += dz_raw.T @ h_prev
            if self.bias: gradbz += dz_raw.sum(axis=0); gradbhz += dz_raw.sum(axis=0)

            gradWr += dr_raw.T @ self.x_list[t]
            gradWhr += dr_raw.T @ h_prev
            if self.bias: gradbr += dr_raw.sum(axis=0); gradbhr += dr_raw.sum(axis=0)

            # Propagate to previous hidden
            delta_h_next = dh_tilde_raw @ self.Whc * r_t + dz_raw @ self.Whz + dr_raw @ self.Whr + delta_h * z_t

            grad_x[:, t, :] = dh_tilde_raw @ self.Wc + dz_raw @ self.Wz + dr_raw @ self.Wr

        # Store gradients
        self.gradWz, self.gradbz, self.gradWhz, self.gradBhz = gradWz, gradbz, gradWhz, gradbhz
        self.gradWr, self.gradbr, self.gradWhr, self.gradBhr = gradWr, gradbr, gradWhr, gradbhr
        self.gradWc, self.gradbc, self.gradWhc, self.gradBhc = gradWc, gradbc, gradWhc, gradbhc

        return grad_x.astype(np.float32)

import numpy as np

from src.structure_layers import ABCStructureLayer


class MaxPoolLayer(ABCStructureLayer):
    def __init__(self, kernel_size: int, stride: int = None):
        """
        :param kernel_size: Размер окна pooling (например, 2 для 2x2)
        :param stride: Шаг перемещения окна. Если None, stride = kernel_size
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.input_shape = None
        self.max_indices = None  # чтобы помнить позиции максимумов для backward

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward pass max pooling.

        :param inputs: Тензор формы (batch, C, H, W)
        :return: Тензор после max pooling
        """
        self.input_shape = inputs.shape
        batch_size, C, H, W = inputs.shape
        KH, KW = self.kernel_size, self.kernel_size
        SH, SW = self.stride, self.stride

        # Вычисляем размеры выхода
        out_H = (H - KH) // SH + 1
        out_W = (W - KW) // SW + 1

        # Инициализация выхода и индексов
        output = np.zeros((batch_size, C, out_H, out_W))
        self.max_indices = np.zeros_like(inputs, dtype=bool)

        # Проходим по каждому окну
        for i in range(out_H):
            for j in range(out_W):
                h_start = i * SH
                w_start = j * SW
                h_end = h_start + KH
                w_end = w_start + KW

                window = inputs[:, :, h_start:h_end, w_start:w_end]
                max_vals = np.max(window, axis=(2, 3))
                output[:, :, i, j] = max_vals

                # Запоминаем позиции максимумов
                max_mask = (window == max_vals[:, :, None, None])
                self.max_indices[:, :, h_start:h_end, w_start:w_end] += max_mask

        return output

    def backward_pass(self, delta: np.ndarray) -> np.ndarray:
        """
        Backward pass для max pooling.

        :param delta: Градиенты со следующего слоя, формы (batch, C, out_H, out_W)
        :return: Градиенты относительно входа, формы (batch, C, H, W)
        """
        if self.input_shape is None or self.max_indices is None:
            raise ValueError("Forward pass must be called before backward_pass.")

        batch_size, C, H, W = self.input_shape
        KH, KW = self.kernel_size, self.kernel_size
        SH, SW = self.stride, self.stride

        grad_input = np.zeros(self.input_shape)

        out_H, out_W = delta.shape[2], delta.shape[3]

        for i in range(out_H):
            for j in range(out_W):
                h_start = i * SH
                w_start = j * SW
                h_end = h_start + KH
                w_end = w_start + KW

                grad_input[:, :, h_start:h_end, w_start:w_end] += delta[:, :, i, j][:, :, None, None] * self.max_indices[:, :, h_start:h_end, w_start:w_end]

        return grad_input


class optMaxPoolLayer(ABCStructureLayer):
    def __init__(self, kernel_size: int, stride: int = None):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.input_shape = None
        self.argmax = None

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward pass (vectorized, im2col)
        """
        self.input_shape = inputs.shape
        N, C, H, W = inputs.shape
        KH, KW = self.kernel_size, self.kernel_size
        SH, SW = self.stride, self.stride

        out_H = (H - KH) // SH + 1
        out_W = (W - KW) // SW + 1

        # Создаем окна с помощью as_strided
        shape = (N, C, out_H, out_W, KH, KW)
        strides = (
            inputs.strides[0],
            inputs.strides[1],
            SH * inputs.strides[2],
            SW * inputs.strides[3],
            inputs.strides[2],
            inputs.strides[3]
        )
        windows = np.lib.stride_tricks.as_strided(inputs, shape=shape, strides=strides, writeable=False)
        windows_reshaped = windows.reshape(N, C, out_H, out_W, KH*KW)

        # Находим максимум и его индекс
        self.argmax = np.argmax(windows_reshaped, axis=-1)
        out = np.max(windows_reshaped, axis=-1)
        return out

    def backward_pass(self, delta: np.ndarray) -> np.ndarray:
        """
        Fully vectorized backward pass (no loops)
        """
        N, C, H, W = self.input_shape
        KH, KW = self.kernel_size, self.kernel_size
        SH, SW = self.stride, self.stride
        out_H, out_W = delta.shape[2], delta.shape[3]

        # Создаем градиенты окон
        d_windows = np.zeros((N, C, out_H, out_W, KH*KW), dtype=delta.dtype)
        np.put_along_axis(d_windows, self.argmax[..., None], delta[..., None], axis=-1)
        d_windows = d_windows.reshape(N, C, out_H, out_W, KH, KW)

        # Инициализация grad_input
        grad_input = np.zeros((N, C, H, W), dtype=delta.dtype)

        # Создаем полностью векторные индексы для добавления градиента
        n_idx = np.arange(N)[:, None, None, None, None, None]
        c_idx = np.arange(C)[None, :, None, None, None, None]
        h_idx = np.arange(out_H)[None, None, :, None, None, None] * SH + np.arange(KH)[None, None, None, None, :, None]
        w_idx = np.arange(out_W)[None, None, None, :, None, None] * SW + np.arange(KW)[None, None, None, None, None, :]

        # Broadcasting всех индексов к форме d_windows
        n_idx = np.broadcast_to(n_idx, d_windows.shape)
        c_idx = np.broadcast_to(c_idx, d_windows.shape)
        h_idx = np.broadcast_to(h_idx, d_windows.shape)
        w_idx = np.broadcast_to(w_idx, d_windows.shape)

        # Аккумулируем градиенты
        np.add.at(grad_input, (n_idx, c_idx, h_idx, w_idx), d_windows)

        return grad_input

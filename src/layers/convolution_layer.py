import numpy as np

from src.layers import ABCLayer
from src.weights_initializers import conv_uniform_init


class ConvLayer(ABCLayer):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int | tuple[int, int] = 3,
            stride: int | tuple[int, int] = 1,
            padding: int | tuple[int, int] = 0,
            bias: bool = True,
            padding_mode: str = 'constant'
        ):
        """
        :param in_channels: Number of channels in the input image.
        :type in_channels: int
        :param out_channels: Number of channels produced by the convolution.
        :type out_channels: int
        :param kernel_size: Size of the convolving kernel.
        :type kernel_size: int | tuple[int]
        :param stride: Stride of the convolution.
        :type stride: int | tuple[int]
        :param padding: Padding added to all four sides of the input.
        :type padding: int | tuple[int]
        :param dilation: Spacing between kernel elements.
        :type dilation: int | tuple[int]
        :param bias: If True, adds a learnable bias to the output.
        :type bias: bool
        :param padding_mode: 'constant' (default) Pads with a zero value. 'edge' Pads with the edge values of array. 'linear_ramp' Pads with the linear ramp between end_value and the array edge value. 'maximum' Pads with the maximum value of all or part of the vector along each axis. 'mean' Pads with the mean value of all or part of the vector along each axis. 'median' Pads with the median value of all or part of the vector along each axis. 'minimum' Pads with the minimum value of all or part of the vector along each axis. 'reflect' Pads with the reflection of the vector mirrored on the first and last values of the vector along each axis. 'symmetric' Pads with the reflection of the vector mirrored along the edge of the array.
        :type padding_mode: str
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.padding_mode = padding_mode
        
        self.H_out: int | None = None
        self.W_out: int | None = None
        self.W, self.b = conv_uniform_init([self.out_channels, self.in_channels, *self.kernel_size], bias)
    
    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        N = inputs.shape[0]
        C_out, _, KH, KW = self.W.shape

        # Добавляем padding
        if any(self.padding):
            inputs_padded = np.pad(
                inputs,
                ((0, 0), (0, 0), self.padding, self.padding),
                mode=self.padding_mode
            )
        else:
            inputs_padded = inputs.copy()

        H_p, W_p = inputs_padded.shape[2], inputs_padded.shape[3]

        # Вычисление размеров выхода
        H_out = (H_p - self.kernel_size[0])//self.stride[0] + 1
        W_out = (W_p - self.kernel_size[1])//self.stride[1] + 1

        # Инициализируем выход
        outputs: np.ndarray = np.zeros((N, C_out, H_out, W_out), dtype=np.float32)
        
        # Свертка
        for n in range(N):  # batch
            for c_out in range(C_out):  # каждый фильтр
                for h in range(H_out):
                    for w in range(W_out):
                        h_start = h*self.stride[0]
                        w_start = w*self.stride[1]

                        # Фрагмент входа, на который накладываем фильтр
                        inputs_window = inputs_padded[
                            n,
                            :,
                            h_start:h_start + KH,
                            w_start:w_start + KW
                        ]

                        # Скалярное произведение
                        outputs[n, c_out, h, w] = np.sum(inputs_window*self.W[c_out])

                if self.b is not None:
                    outputs[n, c_out] += self.b[c_out]
        
        # Сохраняем значения для обучения
        if self.learning:
            self.inputs = inputs.copy()
            self.outputs = outputs.copy()
            self.H_out = H_out
            self.W_out = W_out
        
        return outputs

    def _validate_weights_size(self, weights: np.ndarray):
        error = ''
        if any(self.b):
            if (len(weights[0]) - 1) != len(self.W[0]):
                error = 'Используется смещение, но значение отстуствует.'
        else:
            if len(weights[0]) != len(self.W[0]):
                error = f'Не совпадают размеры весов {len(weights)}x{len(weights[0])} и {len(self.W)}x{len(self.W[0])}.'

        assert len(error) == 0, error

    def get_weights(self) -> np.ndarray:
         return np.column_stack((self.W, self.b)) if any(self.b) else self.W

    def update_weights(self, weights: np.ndarray):
        # Валидируем веса
        self._validate_weights_size(weights)

        # Обновляем веса
        if any(self.b):
            self.W, self.b = weights[:, :-1], weights[:, -1]
        else:
            self.W = weights
    
    def backward_pass(self, delta: np.ndarray) -> np.ndarray:
        """
        :param delta: Локальная ошибка с правого слоя (shape: [n, c_out, h, w]).
        :type delta: np.ndarray
        """
        KH, KW = self.kernel_size
        N, C_out, H_out, W_out = delta.shape

        # Добавляем padding
        if any(self.padding):
            inputs_padded = np.pad(
                self.inputs,
                ((0, 0), (0, 0), self.padding, self.padding),
                mode=self.padding_mode
            )
        else:
            inputs_padded = self.inputs.copy()

        delta_padded = np.zeros_like(inputs_padded, dtype=np.float32)
        self.gradW = np.zeros_like(self.W, dtype=np.float32)
        self.gradb = np.zeros_like(self.b, dtype=np.float32) if self.b is not None else None

        # Вычисляем градиенты
        for n in range(N):
            for c_out in range(C_out):
                for h in range(H_out):
                    for w in range(W_out):
                        h_start = h * self.stride[0]
                        w_start = w * self.stride[1]
                        inputs_window = inputs_padded[n, :, h_start:h_start + KH, w_start:w_start + KW]

                        # Градиент по весам
                        self.gradW[c_out] += delta[n, c_out, h, w]*inputs_window

                        # Градиент по входу
                        delta_padded[n, :, h_start:h_start + KH, w_start:w_start + KW] += delta[n, c_out, h, w]*self.W[c_out]

                # Градиент по bias
                if self.b is not None:
                    self.gradb[c_out] += np.sum(delta[n, c_out, :, :])

        # Убираем padding
        if any(self.padding):
            delta = delta_padded[:, :, self.padding[0]:-self.padding[1], self.padding[0]:-self.padding[1]]
        else:
            delta = delta_padded

        return delta

    def _im2col_indices(self, inputs):
        """
        x: padded input, shape (N, C, H, W)
        returns cols: shape (N, C*KH*KW, out_h*out_w)
        """
        N, C, H, W = inputs.shape
        KH, KW = self.kernel_size
        out_h = (H - KH) // self.stride[0] + 1
        out_w = (W - KW) // self.stride[1] + 1

        # индексы внутри ядра
        i0 = np.repeat(np.arange(KH), KW)
        i0 = np.tile(i0, C)                     # (C*KH*KW,)
        j0 = np.tile(np.arange(KW), KH)
        j0 = np.tile(j0, C)                     # (C*KH*KW,)
        # смещения по выходным позициям
        i1 = self.stride[0] * np.repeat(np.arange(out_h), out_w)   # (out_h*out_w,)
        j1 = self.stride[1] * np.tile(np.arange(out_w), out_h)     # (out_h*out_w,)

        # итоговые координаты
        i = i0.reshape(-1, 1) + i1.reshape(1, -1)  # (C*KH*KW, out_h*out_w)
        j = j0.reshape(-1, 1) + j1.reshape(1, -1)  # (C*KH*KW, out_h*out_w)

        k = np.repeat(np.arange(C), KH * KW).reshape(-1, 1)  # (C*KH*KW, 1)

        # x[:, k, i, j] -> (N, C*KH*KW, out_h*out_w)
        cols = inputs[:, k, i, j]
        return cols  # shape (N, C*KH*KW, out_h*out_w)


    def _optimized_forward(self, inputs):
        """
        Векторизированный forward Conv2D через im2col.

        x: вход (N, C_in, H_in, W_in)

        Возвращает: out (N, C_out, H_out, W_out)
        """
        if self.learning: self.inputs = inputs.copy()

        N = inputs[0]
        C_out, _, KH, KW = self.W.shape

        # pad
        if any(self.padding):
            x_padded = np.pad(inputs, ((0,0), (0,0), self.padding, self.padding), mode='constant')
        else:
            x_padded = inputs

        H_p, W_p = x_padded.shape[2], x_padded.shape[3]
        H_out = (H_p - KH) // self.stride[0] + 1
        W_out = (W_p - KW) // self.stride[1] + 1

        # im2col: (N, K, L) where K = C_in*KH*KW, L = H_out*W_out
        cols = self._im2col_indices(x_padded)  # note: x already padded
        N_, K, L = cols.shape
        assert N_ == N

        # reshape веса в матрицу (C_out, K)
        W_col = self.W.reshape(C_out, -1)  # (C_out, K)

        # матричное умножение: для каждого примера выполним W_col @ cols[n]
        # используем tensordot, чтобы избежать явного цикла по батчу:
        # tensordot(cols, W_col, axes=([1],[1])) -> (N, L, C_out)
        out_nlc = np.tensordot(cols, W_col, axes=([1], [1]))  # (N, L, C_out)
        out = out_nlc.transpose(0, 2, 1).reshape(N, C_out, H_out, W_out)  # (N, C_out, H_out, W_out)

        if self.b is not None:
            out += self.b.reshape(1, -1, 1, 1)

        return out


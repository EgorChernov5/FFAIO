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
    
    def backward_pass(self, delta: np.ndarray) -> np.ndarray:
        """
        :param delta: Локальная ошибка с правого слоя (shape: (n, c_out, h, w)).
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


class optConvLayer(ABCLayer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int] = 3,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        bias: bool = True,
        padding_mode: str = "constant",
        clip_value: float = 1e-3
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.padding_mode = padding_mode

        self.W, self.b = conv_uniform_init(
            [self.out_channels, self.in_channels, *self.kernel_size], bias
        )
        self.W = self.W.astype(np.float32)
        if self.b is not None:
            self.b = self.b.astype(np.float32)

        self.H_out = None
        self.W_out = None
        self.clip_value = clip_value  # <-- максимум для clip

    def _im2col(self, inputs):
        N, C, H, W = inputs.shape
        KH, KW = self.kernel_size
        SH, SW = self.stride
        PH, PW = self.padding

        # padding
        x = np.pad(inputs, ((0, 0), (0, 0), (PH, PH), (PW, PW)), mode='constant')

        H_p, W_p = x.shape[2:]
        H_out = (H_p - KH)//SH + 1
        W_out = (W_p - KW)//SW + 1

        # индексы
        i0 = np.repeat(np.arange(KH), KW)
        i0 = np.tile(i0, C)

        j0 = np.tile(np.arange(KW), KH)
        j0 = np.tile(j0, C)

        i1 = SH * np.repeat(np.arange(H_out), W_out)
        j1 = SW * np.tile(np.arange(W_out), H_out)

        i = i0.reshape(1, -1) + i1.reshape(-1, 1)
        j = j0.reshape(1, -1) + j1.reshape(-1, 1)
        k = np.repeat(np.arange(C), KH*KW).reshape(1, -1)

        cols = x[:, k, i, j]   # (N, H_out*W_out, C*KH*KW)
        return cols

    def _col2im(self, cols, input_shape):
        N, C, H, W = input_shape
        KH, KW = self.kernel_size
        SH, SW = self.stride
        PH, PW = self.padding

        H_outW_out = cols.shape[1]
        H_out = (H + 2*PH - KH)//SH + 1
        W_out = (W + 2*PW - KW)//SW + 1

        x_padded = np.zeros((N, C, H + 2*PH, W + 2*PW), dtype=cols.dtype)

        i0 = np.repeat(np.arange(KH), KW)
        i0 = np.tile(i0, C)
        j0 = np.tile(np.arange(KW), KH)
        j0 = np.tile(j0, C)
        k  = np.repeat(np.arange(C), KH*KW)

        i1 = SH * np.repeat(np.arange(H_out), W_out)
        j1 = SW * np.tile(np.arange(W_out), H_out)

        i = i0.reshape(1,-1) + i1.reshape(-1,1)
        j = j0.reshape(1,-1) + j1.reshape(-1,1)

        cols_reshaped = cols.reshape(N, H_out*W_out, C*KH*KW)

        for n in range(N):
            np.add.at(x_padded[n], (k, i, j), cols_reshaped[n])

        return x_padded[:, :, PH:H+PH, PW:W+PW]

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        N = inputs.shape[0]
        C_out, C_in, KH, KW = self.W.shape

        cols = self._im2col(inputs)  # (N, H_out*W_out, C*KH*KW)
        W_col = self.W.reshape(C_out, -1)  # (C_out, C*KH*KW)

        out = cols @ W_col.T  # (N, H_out*W_out, C_out)

        if self.b is not None:
            out += self.b.reshape(1, 1, -1)

        # reshape к PyTorch формату
        PH, PW = self.padding
        SH, SW = self.stride
        H_out = (inputs.shape[2] + 2*PH - KH)//SH + 1
        W_out = (inputs.shape[3] + 2*PW - KW)//SW + 1

        outputs = out.transpose(0, 2, 1).reshape(N, C_out, H_out, W_out)

        # Сохраняем значения для обучения
        if self.learning:
            self.inputs = inputs.copy()
            self.outputs = outputs.copy()
            self.H_out = H_out
            self.W_out = W_out

        return outputs

    def backward_pass(self, delta: np.ndarray) -> np.ndarray:
        # delta: (N, C_out, H_out, W_out)
        N, C_out, H_out, W_out = delta.shape
        C_out, C_in, KH, KW = self.W.shape
        PH, PW = self.padding
        SH, SW = self.stride

        # ---- (1) im2col для входа ----
        X_col = self._im2col(self.inputs)
        # X_col: (N, H_out*W_out, C_in*KH*KW)

        # ---- (2) delta reshape ----
        delta_col = delta.reshape(N, C_out, -1).transpose(0,2,1)  # (N, H_out*W_out, C_out)

        # ---- (3) gradW ----
        # суммируем по batch
        self.gradW = (delta_col.transpose(0,2,1) @ X_col).sum(0)
        self.gradW = self.gradW.reshape(C_out, C_in, KH, KW)

        # ---- (4) gradb ----
        if self.b is not None:
            self.gradb = delta.sum(axis=(0,2,3))

        # ---- (5) grad_input (через col2im) ----
        W_col = self.W.reshape(C_out, -1)
        delta_in_col = delta_col @ W_col  # (N, H_out*W_out, C_in*KH*KW)

        # col2im
        delta_in = self._col2im(delta_in_col,
                                self.inputs.shape)

        return delta_in
    
    def to_str(self):
        return 'ConvLayer'

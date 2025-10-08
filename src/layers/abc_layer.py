from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod

from src.losses import Loss
from src.activations import ActivationFunction


class Layer(ABC):
    def __init__(self, in_features: int, out_features: int, activation_function: ActivationFunction):
        self.in_features = in_features
        self.out_features = out_features

        self.activation_function = activation_function

    @abstractmethod
    def get_size(self) -> list[int]:
        raise NotImplementedError()

    @abstractmethod
    def len_weights(self) -> int:
        raise NotImplementedError()
    
    @abstractmethod
    def get_weights(self) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def update_weights(self):
        raise NotImplementedError()

    @abstractmethod
    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
    
    @abstractmethod
    def gradient(self, loss: Loss, prev_layer: "Layer" | None = None) ->  np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def partial_derivative_wrt_w(self):
        raise NotImplementedError()

    @abstractmethod
    def partial_derivative_wrt_a(self):
        raise NotImplementedError()
    
    # TODO
    # def partial_derivative_loss_wrt_a(self, i_neuron: int, input: float, loss: Loss, prev_layer: "Layer" | None) -> float:
    #     dL_da = 0.
    #     # Проверяем есть ли предыдущий слой, если нет, то считаем dL_da для выходного слоя
    #     if prev_layer is None:
    #         # Для каждого веса выходного нейрона частная производная Loss и активации равна одному и тому же числу
    #         dL_da = loss.partial_derivative_wrt_a(i_neuron, input)
    #     else:
    #         # Приращение каждого веса нейрона влияет на приращение Loss по каждому выходу
    #         # поэтому dL_da^(n-1) = dL_da^(n) * da_dz^(n) * dz^(n)_da^(n-1)
    #         for i_output_neuron in range(prev_layer.out_features):
    #             dL_da_p = loss.dL_dA[i_output_neuron]
    #             da_dz_p = prev_layer.activation_function.dA_dZ[i_output_neuron]
    #             dzp_da = prev_layer.partial_derivative_wrt_a(i_output_neuron, i_neuron)

    #             dL_da += dL_da_p*da_dz_p*dzp_da

    #     return dL_da
    def partial_derivative_loss_wrt_a(self, i_neuron: int, inputs: np.ndarray, loss: Loss, prev_layer: "Layer" | None) -> float:
        dL_da = 0.
        # Проверяем есть ли предыдущий слой, если нет, то считаем dL_da для выходного слоя
        if prev_layer is None:
            # Для каждого веса выходного нейрона частная производная Loss и активации равна одному и тому же числу
            dL_da = loss.partial_derivative_wrt_a(i_neuron, inputs)
        else:
            # TODO
            # Приращение каждого веса нейрона влияет на приращение Loss по каждому выходу
            # поэтому dL_da^(n-1) = dL_da^(n) * da_dz^(n) * dz^(n)_da^(n-1)
            for i_output_neuron in range(prev_layer.out_features):
                dL_da_p = loss.dL_dA[i_output_neuron]
                da_dz_p = prev_layer.activation_function.dA_dZ[i_output_neuron]
                dzp_da = prev_layer.partial_derivative_wrt_a(i_output_neuron, i_neuron)

                dL_da += dL_da_p*da_dz_p*dzp_da

        return dL_da

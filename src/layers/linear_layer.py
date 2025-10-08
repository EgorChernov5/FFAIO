import numpy as np

from src.layers import Layer
from src.activations import ActivationFunction
from src.nodes import Neuron
from src.losses import Loss


class LinearLayer(Layer):
    def __init__(self, in_features: int, out_features: int, activation_function: ActivationFunction, bias: bool):
        super().__init__(in_features, out_features, activation_function)

        self.bias = bias
        self.neurons = [Neuron(self.in_features, self.bias) for _ in range(self.out_features)]

    def get_size(self) -> list[int]:
        return [self.out_features, self.in_features + 1 if self.bias else self.in_features]

    def len_weights(self) -> int:
        size = self.get_size()
        return size[0]*size[1]
    
    def get_weights(self) -> np.ndarray:
        return np.array([neuron.get_weights() for neuron in self.neurons])

    def update_weights(self, weights: np.ndarray):
        for neuron, weights_neuron in zip(self.neurons, weights):
            neuron.update_weights(weights_neuron)

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        activations = np.array([neuron(inputs) for neuron in self.neurons])
        # Транспонируем, чтобы строки были объектами, а столбцы выходными значениями нейронов
        return self.activation_function(activations.T)
    
    def gradient(self, loss: Loss, prev_layer: Layer | None = None) -> np.ndarray:
        # Получаем размеры слоя (кол-во нейронов, кол-во весов на каждом нейроне)
        layer_size = self.get_size()
        # Формируем хранение частных производных
        dL_dW = np.zeros(layer_size)
        dL_dA = np.zeros(layer_size[0])
        dA_dZ = np.zeros(layer_size[0])
        # Считаем частные производные Loss функции по весам каждого нейрона
        for i_neuron in range(layer_size[0]):
            # Приращение Loss по каждому выходу
            dL_da = self.partial_derivative_loss_wrt_a(i_neuron, self.activation_function.A, loss, prev_layer)
            # Частная производная функции активации по выходу ф-и линейной трансформации
            da_dz = self.activation_function.partial_derivative_wrt_z(i_neuron)
            # TODO
            # Записываем частную производную Loss по активации и активации по весу
            dL_dA[i_neuron] = dL_da
            dA_dZ[i_neuron] = da_dz
            # Считаем частные производные ф-и линейной трансформации по каждому весу нейрона
            for i_weight in range(layer_size[1]):
                # Если считаем по bias, то dz_dw будет равна 1
                dz_dw = self.partial_derivative_wrt_w(i_neuron, i_weight)
                # Частная производная Loss по весу нейрона: dL_dw = dL_da*da_dz*dz_dw
                dL_dW[i_neuron, i_weight] = dL_da*da_dz*dz_dw

        # Сохраняем частные производные
        loss.dL_dA = dL_dA
        self.activation_function.dA_dZ = dA_dZ
        
        return dL_dW
    
    def partial_derivative_wrt_w(self, i_neuron: int, i_weight: int) -> float:
        return self.neurons[i_neuron].partial_derivative_wrt_w(i_weight)
    
    def partial_derivative_wrt_a(self, i_neuron: int, i_feature: int) -> float:
        return self.neurons[i_neuron].partial_derivative_wrt_a(i_feature)

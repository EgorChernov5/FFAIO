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
        # Транспонируем, чтобы строки были объектами, а столбцы выходными значениями нейронов
        activations = np.array([neuron(inputs) for neuron in self.neurons]).T
        # Прогоняем выходны нейронов через ф-ю активации
        return self.activation_function(activations)
    
    def gradient(self, loss: Loss, prev_layer: Layer | None = None) -> np.ndarray:
        # Получаем размеры слоя (кол-во нейронов, кол-во весов на каждом нейроне)
        n_neurons, n_weights = self.get_size()
        # Формируем хранение частных производных
        grad_W = np.zeros([n_neurons, n_weights])
        dL_dA = np.empty((0, len(self.activation_function.A)))
        # Считаем частные производные Loss функции по весам каждого нейрона
        for i_neuron in range(n_neurons):
            # Приращение Loss по каждому выходу
            dL_da = self.partial_derivative_loss_wrt_a(i_neuron, self.activation_function.A, loss, prev_layer)
            # Частная производная функции активации по выходу ф-и линейной трансформации
            da_dz = self.activation_function.partial_derivative_wrt_z(i_neuron)
            # Записываем частную производную Loss по активации и активации по весу
            dL_dA = np.vstack((dL_dA, dL_da))
            # Считаем частные производные ф-и линейной трансформации по каждому весу нейрона
            for i_weight in range(n_weights):
                # Если считаем по bias, то dz_dw будет равна 1
                dz_dw = self.partial_derivative_wrt_w(i_neuron, i_weight)
                # Частная производная Loss по весу нейрона: dL_dw = dL_da*da_dz*dz_dw
                grad_W[i_neuron, i_weight] = np.mean(dL_da*da_dz*dz_dw)

        # Сохраняем частные производные
        loss.dL_dA = dL_dA
        
        return grad_W
    
    def partial_derivative_wrt_w(self, i_neuron: int, i_weight: int) -> np.ndarray:
        return self.neurons[i_neuron].partial_derivative_wrt_w(i_weight)
    
    def partial_derivative_wrt_a(self, i_neuron: int, i_feature: int) -> float:
        return self.neurons[i_neuron].partial_derivative_wrt_a(i_feature)

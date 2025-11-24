import unittest
import numpy as np

from src.optimizers import SGD

class DummyLayer:
    def __init__(self, weights, grad):
        self._weights = np.array(weights, dtype=float)
        self._grad = np.array(grad, dtype=float)
        self.updated_weights = None
        self.gradient_args = []

    def get_weights(self):
        return self._weights.copy()

    def update_weights(self, new_weights):
        self.updated_weights = new_weights.copy()
        self._weights = new_weights.copy()

    def gradient(self, loss, prev_layer=None):
        self.gradient_args.append((loss, prev_layer))
        return self._grad.copy()

class DummyLoss:
    pass

class TestSGDOptimizer(unittest.TestCase):
    def test_step_updates_weights_for_all_layers(self):
        l1 = DummyLayer([1.0, 2.0], [0.1, 0.2])
        l2 = DummyLayer([3.0, 4.0], [0.3, 0.4])
        arch = [l1, l2]
        loss = DummyLoss()
        sgd = SGD(batch_size=1, lr=0.5)
        sgd.arch_model = arch
        sgd.loss = loss

        sgd.step()

        np.testing.assert_allclose(l2.updated_weights, [2.85, 3.8])
        np.testing.assert_allclose(l1.updated_weights, [0.95, 1.9])

    def test_gradient_called_with_correct_args(self):
        l1 = DummyLayer([1, 2], [0.1, 0.2])
        l2 = DummyLayer([3, 4], [0.3, 0.4])
        arch = [l1, l2]
        loss = DummyLoss()
        sgd = SGD(batch_size=1, lr=0.1)
        sgd.arch_model = arch
        sgd.loss = loss

        sgd.step()

        self.assertIs(l2.gradient_args[0][0], loss)
        self.assertIs(l2.gradient_args[0][1], None)
        self.assertIs(l1.gradient_args[0][0], loss)
        self.assertIs(l1.gradient_args[0][1], l2)

    def test_step_chain_of_three_layers(self):
        l1 = DummyLayer([1, 1], [0.1, 0.1])
        l2 = DummyLayer([2, 2], [0.2, 0.2])
        l3 = DummyLayer([3, 3], [0.3, 0.3])
        arch = [l1, l2, l3]
        sgd = SGD(batch_size=1, lr=1.0)
        sgd.arch_model = arch
        sgd.loss = DummyLoss()
        sgd.step()
        np.testing.assert_allclose(l3.updated_weights, [2.7, 2.7])
        np.testing.assert_allclose(l2.updated_weights, [1.8, 1.8])
        np.testing.assert_allclose(l1.updated_weights, [0.9, 0.9])


if __name__ == "__main__":
    unittest.main()

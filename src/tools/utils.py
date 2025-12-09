from pathlib import Path
import numpy as np

import torch
from torch.nn import Module

from src.models import ABCModel


def to_one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    """
    Преобразует вектор меток классов в one-hot encoding.

    :param y: Вектор меток классов размерности (n_batch,), значения от 0 до n_classes-1.
    :type y: np.ndarray
    :param n_classes: Общее количество классов.
    :type n_classes: int

    :return: One-hot представление меток размерности (n_batch, n_classes).
    :rtype: np.ndarray
    """
    y = np.asarray(y, dtype=int)
    one_hot = np.zeros((y.shape[0], n_classes), dtype=np.float32)
    one_hot[np.arange(y.shape[0]), y] = 1.0
    return one_hot


def share_weights(src_model: ABCModel | Module, dst_model: ABCModel | Module):
    weights = {}
    src_model.eval()
    with torch.no_grad():
        if isinstance(src_model, Module) and isinstance(dst_model, ABCModel):
            for name, param in src_model.named_parameters():
                name_layer, name_weight = name.split('.')
                weights[name_layer] = {name_weight: param.detach().numpy()} if name_weight == 'weight' else weights[name_layer] | {name_weight: param.detach().numpy()}

            for layer, new_weights in zip(dst_model.get_weights_layers(), weights.values()):
                new_W, new_b = new_weights.values()
                layer.update_weights(new_W, new_b)
        elif isinstance(src_model, ABCModel) and isinstance(dst_model, Module):
            new_state_dict = src_model.get_weights()
            new_state_dict = {k: torch.from_numpy(v) for k, v in new_state_dict.items()}
            dst_model.load_state_dict(new_state_dict)


def save_weights(custom_model: ABCModel, pytorch_model: Module, path_weights: str | Path):
    custom_model.eval()
    share_weights(custom_model, pytorch_model)
    with torch.no_grad():
        torch.save(pytorch_model.state_dict(), path_weights)


def load_weights(custom_model: ABCModel, pytorch_model: Module, path_weights: str | Path):
    custom_model.eval()
    with torch.no_grad():
        pytorch_model.load_state_dict(torch.load(path_weights, weights_only=True))
    share_weights(pytorch_model, custom_model)



import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from src.models import ABCModel


def vis_losses_metrics(model: ABCModel):
    # Строим графики
    plt.figure(figsize=(10, 4))

    # --- График потерь ---
    plt.subplot(1, 2, 1)
    plt.plot(model.losses_train, label='Train loss')
    plt.plot(model.losses_val, label='Val loss')
    plt.xlabel('Эпоха')
    plt.ylabel('Loss')
    plt.title('Изменение функции потерь')
    plt.legend()
    plt.grid(True)

    # --- График точности ---
    plt.subplot(1, 2, 2)
    plt.plot(model.metrics_val, label='Val accuracy')
    plt.xlabel('Эпоха')
    plt.ylabel('Accuracy')
    plt.title('Изменение точности')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def vis_roc_curve(y_test: np.ndarray, y_prob: np.ndarray):
    # Вычисляем координаты ROC-кривой
    roc_y_true = y_test.argmax(axis=1) if y_test.shape[1] == 2 else y_test.squeeze()
    roc_y_prob = y_prob[:, 1] if y_prob.shape[1] == 2 else y_prob.squeeze()
    fpr, tpr, thresholds = roc_curve(roc_y_true, roc_y_prob)
    # Площадь под кривой (AUC)
    roc_auc = auc(fpr, tpr)

    # Визуализация ROC-кривой
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC-кривая (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Случайная модель')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC-кривая')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

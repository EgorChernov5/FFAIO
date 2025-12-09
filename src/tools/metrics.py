from typing import Callable, Any, Tuple
import numpy as np

from sklearn.manifold import TSNE

from src.models import ABCModel
from src.data_loaders import ABCLoader


def collect_predictions(
        loader: ABCLoader,
        X: np.ndarray, y: np.ndarray,
        model: ABCModel,
        postprocess: Callable[[np.ndarray], np.ndarray] | None = None,
        return_scores: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Сбор истинных меток и предсказаний модели по батчам.
    
    :param loader: Источник данных, возвращающий батчи.
    :type loader: ABCLoader
    :param model: Модель, реализующая методы __call__ или predict_proba.
    :type model: ABCModel
    :param postprocess: Функция по обработке вероятностей.
    :type postprocess: Callable[[np.ndarray], np.ndarray] | None
    :param return_scores: Если True — возвращает матрицу вероятностей.
    :type return_scores: bool

    :return: y_true, y_pred и, опционально, y_scores.
    :rtype: tuple[np.ndarray, np.ndarray, np.ndarray | None]
    """
    y_true_all = []
    y_pred_all = []
    y_score_all = [] if return_scores else None

    for _, X_batch, y_batch in loader.get_data(X, y):
        if return_scores:
            scores = model(X_batch)
            preds = np.argmax(scores, axis=1) if postprocess is None else postprocess(scores)
            y_score_all.append(scores)
        else:
            preds = model(X_batch, postprocess)

        y_true_all.append(y_batch)
        y_pred_all.append(preds)

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)

    if return_scores:
        y_scores = np.vstack(y_score_all)
        return y_true, y_pred, y_scores
    else:
        return y_true, y_pred, None


# -----------------------------------------------------------
# 1. Подсчёт матрицы неточностей
# -----------------------------------------------------------
def confusion_matrix_multiclass(y_true, y_pred, num_classes=None):
    if num_classes is None:
        num_classes = max(y_true.max(), y_pred.max()) + 1

    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def confusion_matrix_stream(
        loader: ABCLoader,
        X, y,
        model: ABCModel,
        num_classes: int,
        postprocess: Callable[[np.ndarray], np.ndarray] | None = None
    ) -> np.ndarray:
    """
    Подсчёт матрицы неточностей по батчам.

    :param loader: Источник данных, который предоставляет батчи.
    :type loader: ABCLoader
    :param model: Модель, возвращающая предсказанные классы.
    :type model: ABCModel
    :param num_classes: Количество классов.
    :type num_classes: int
    :param postprocess: Функция по обработке вероятностей.
    :type postprocess: Callable[[np.ndarray], np.ndarray] | None

    :return: Матрица неточностей размера [num_classes, num_classes].
    :rtype: np.ndarray
    """
    cm = np.zeros((num_classes, num_classes), dtype=int)

    for _, X_batch, y_batch in loader.get_data(X, y):
        preds = model(X_batch, postprocess).astype(int)
        for t, p in zip(y_batch, preds):
            cm[t, p] += 1

    return cm


# -----------------------------------------------------------
# 2. Метрики accuracy, precision, recall, F1
# -----------------------------------------------------------
def accuracy_multiclass(cm):
    return np.trace(cm) / np.sum(cm)


def precision_multiclass(cm):
    # Precision_k = TP_k / (TP_k + FP_k)
    TP = np.diag(cm)
    FP = np.sum(cm, axis=0) - TP
    return TP / (TP + FP + 1e-12)


def recall_multiclass(cm):
    # Recall_k = TP_k / (TP_k + FN_k)
    TP = np.diag(cm)
    FN = np.sum(cm, axis=1) - TP
    return TP / (TP + FN + 1e-12)


def f1_multiclass(cm):
    p = precision_multiclass(cm)
    r = recall_multiclass(cm)
    return 2 * p * r / (p + r + 1e-12)


# -----------------------------------------------------------
# Macro / Micro / Weighted агрегирование
# -----------------------------------------------------------
def macro_avg(metric_per_class):
    return np.mean(metric_per_class)


def weighted_avg(metric_per_class, support):
    return np.sum(metric_per_class * support) / np.sum(support)


def micro_f1(cm):
    # micro precision = micro recall = accuracy
    acc = accuracy_multiclass(cm)
    return acc  # micro F1 = micro Precision = micro Recall


def roc_curve_binary(
        y_true_bin: np.ndarray, 
        scores: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Построение ROC-кривой для бинарной задачи.
    
    :param y_true_bin: Истинные бинарные метки класса.
    :type y_true_bin: np.ndarray
    :param scores: Вероятности положительного класса.
    :type scores: np.ndarray

    :return: fpr, tpr и пороги.
    :rtype: tuple[np.ndarray, np.ndarray, np.ndarray]
    """
    thresholds = np.sort(np.unique(scores))[::-1]
    tprs, fprs = [], []

    for t in thresholds:
        pred = (scores >= t).astype(int)

        tp = np.sum((y_true_bin == 1) & (pred == 1))
        tn = np.sum((y_true_bin == 0) & (pred == 0))
        fp = np.sum((y_true_bin == 0) & (pred == 1))
        fn = np.sum((y_true_bin == 1) & (pred == 0))

        tpr = tp / (tp + fn + 1e-12)
        fpr = fp / (fp + tn + 1e-12)

        tprs.append(tpr)
        fprs.append(fpr)

    return np.array(fprs), np.array(tprs), thresholds


def auc_score(fprs: np.ndarray, tprs: np.ndarray) -> float:
    """
    Вычисление площади под ROC-кривой.

    :param fprs: Массив значений false positive rate.
    :type fprs: np.ndarray
    :param tprs: Массив значений true positive rate.
    :type tprs: np.ndarray

    :return: Значение AUC.
    :rtype: float
    """
    return np.trapezoid(tprs, fprs)


def roc_auc_multiclass(
        y_true: np.ndarray, 
        y_scores: np.ndarray, 
        num_classes: int | None = None
    ) -> dict[int, dict[str, np.ndarray | float]]:
    """
    ROC-AUC для мультиклассовой классификации методом one-vs-rest.

    :param y_true: Истинные метки класса [n_samples].
    :type y_true: np.ndarray
    :param y_scores: Вероятности классов [n_samples, n_classes].
    :type y_scores: np.ndarray
    :param num_classes: Количество классов.
    :type num_classes: int | None

    :return: Для каждого класса: fprs, tprs, thresholds, auc.
    :rtype: dict[int, dict[str, np.ndarray | float]]
    """
    if num_classes is None:
        num_classes = y_scores.shape[1]

    roc_data = {}

    for c in range(num_classes):
        y_true_bin = (y_true == c).astype(int)
        scores = y_scores[:, c]

        fprs, tprs, th = roc_curve_binary(y_true_bin, scores)
        auc = auc_score(fprs, tprs)

        roc_data[c] = {
            "fprs": fprs,
            "tprs": tprs,
            "thresholds": th,
            "auc": auc,
        }

    return roc_data


def evaluate_multiclass_loader(
        loader: ABCLoader,
        X, y,
        model: ABCModel,
        num_classes: int,
        return_scores: bool = False
    ) -> dict[str, Any]:
    """
    Полная оценка мультиклассовой модели с батчевой обработкой.

    :param loader: Источник данных, возвращающий батчи X, y.
    :type loader: ABCLoader
    :param model: Модель, возвращающая классы или вероятности.
    :type model: ABCModel
    :param num_classes: Количество классов.
    :type num_classes: int
    :param return_scores: Если True, будут вычислены ROC и AUC.
    :type return_scores: bool

    :return: Словарь со всеми метриками.
    :rtype: dict[str, Any]
    """
    y_true, y_pred, y_scores = collect_predictions(
        loader, X, y, model, return_scores=return_scores
    )

    cm = confusion_matrix_multiclass(y_true, y_pred, num_classes)
    support = np.sum(cm, axis=1)

    prec = precision_multiclass(cm)
    rec  = recall_multiclass(cm)
    f1   = f1_multiclass(cm)

    metrics = {
        "confusion_matrix": cm,
        "accuracy": accuracy_multiclass(cm),

        "precision_per_class": prec,
        "recall_per_class": rec,
        "f1_per_class": f1,

        "macro_precision": macro_avg(prec),
        "macro_recall": macro_avg(rec),
        "macro_f1": macro_avg(f1),

        "weighted_precision": weighted_avg(prec, support),
        "weighted_recall": weighted_avg(rec, support),
        "weighted_f1": weighted_avg(f1, support),

        "micro_f1": micro_f1(cm)
    }

    if return_scores:
        metrics["roc_auc"] = roc_auc_multiclass(y_true, y_scores)

    return metrics


def tsne_batch(
        loader: ABCLoader,
        X: np.ndarray,
        y: np.ndarray,
        model,  # класс с методом get_features(X_batch) -> np.ndarray
        max_samples: int = 2000,
        n_components: int = 2,
        perplexity: float = 30.0,
        learning_rate: float = 200.0,
        n_iter: int = 1000,
        random_state: int | None = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    t-SNE для большого датасета, обработка батчами через BaseLoader.
    
    :param loader: Загружчик данных с методом get_data(X, y), возвращающий батчи.
    :type loader: BaseLoader
    :param X: Полный массив признаков.
    :type X: np.ndarray
    :param y: Полный массив меток.
    :type y: np.ndarray
    :param model: Модель с методом get_features(X_batch), возвращающим признаки [n_batch, n_features].
    :type model: Any
    :param max_samples: Максимальное количество сэмплов для t-SNE.
    :type max_samples: int
    :param n_components: Размерность выходного пространства (2 или 3).
    :type n_components: int
    :param perplexity: Параметр perplexity t-SNE.
    :type perplexity: float
    :param learning_rate: Скорость обучения.
    :type learning_rate: float
    :param n_iter: Количество итераций оптимизации.
    :type n_iter: int
    :param random_state: Начальное состояние генератора для воспроизводимости.
    :type random_state: int | None

    :return: Эмбеддинги t-SNE и соответствующие метки [n_samples_selected, n_components], [n_samples_selected].
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    features_list = []
    labels_list = []
    total_samples = 0

    # Получаем признаки и метки батчами
    for _, X_batch, y_batch in loader.get_data(X, y):
        feats = model.get_features(X_batch)
        features_list.append(feats)
        labels_list.append(y_batch)
        total_samples += feats.shape[0]
        if total_samples >= max_samples:
            break

    # Объединяем все батчи
    X_all = np.vstack(features_list)
    y_all = np.concatenate(labels_list)

    # Случайная выборка max_samples
    if X_all.shape[0] > max_samples:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(X_all.shape[0], size=max_samples, replace=False)
        X_all = X_all[idx]
        y_all = y_all[idx]

    # t-SNE
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        learning_rate=learning_rate,
        max_iter=n_iter,
        init="random",
        random_state=random_state,
        metric="euclidean"
    )

    X_embedded = tsne.fit_transform(X_all)
    return X_embedded, y_all

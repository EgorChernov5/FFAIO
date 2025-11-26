import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def load_mushroom_dataset(labels: list[int] = [1, 0], targets_shape: int = 2, random_state: int | None = None) -> tuple[np.ndarray]:
    """
    Функция возвращает тренировочный, валидационный и тестировочный набор данных.

    :param labels: Список с метками (тк бинарная классификация, то [1, 0] или [1, -1]).
    :type labels: list[int]
    :param targets_shape: Кол-во меток объекта (может быть 1 или 2).
    :type targets_shape: list[int]
    :param random_state: Значение random_state.
    :type random_state: int | None

    :return: Кортеж с наборами данных (X_train, X_val, X_test, y_train, y_val, y_test).
    :rtype: tuple[np.ndarray]
    """
    # Загружаем данные
    mushroom = fetch_ucirepo(id=73)

    # Получаем признаки и метки
    X = mushroom.data.features
    y = mushroom.data.targets

    # Предобрабатываем данные
    # Инструменты для предобработки
    label_encoder = LabelEncoder()
    scaler = StandardScaler()
    # poisonous=labels[0], edible=labels[1]
    y = np.where(y == 'p', labels[0], labels[1])
    if targets_shape == 2:
        y = np.eye(targets_shape)[y].squeeze()
    # Кодируем все признаки
    cols = X.keys()
    for col in cols:
        new_col = str(col) + "_n"
        X[new_col] = label_encoder.fit_transform(X[col])
        del X[col]

    # Разделяем на выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, shuffle=True, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=random_state, shuffle=True, stratify=y_train)

    # Скелим данные
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.fit_transform(X_val)
    X_test = scaler.fit_transform(X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test
import os
from pathlib import Path
from ucimlrepo import fetch_ucirepo
import requests
import zipfile

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

import torch
import torchvision

from src.tools import utils


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
        y = np.where(y == 1, 1, 0)
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
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test


def load_mnist_dataset(save_path: str | Path, val_size: float | None = None, random_state: int | None = None) -> tuple[np.ndarray]:
    """
    Функция возвращает тренировочный, валидационный и тестировочный набор данных.

    :param save_path: Путь куда сохранить датасеты.
    :type save_path: str | Path
    :param val_size: Доля валидационной выборки от тренировочной.
    :type val_size: float | None
    :param random_state: Значение random_state.
    :type random_state: int | None

    :return: Кортеж с наборами данных (X_train, X_val, X_test, y_train, y_val, y_test).
    :rtype: tuple[np.ndarray]
    """
    # Загружаем данные
    train_ds = torchvision.datasets.MNIST(root=save_path, train=True, download=True)
    test_ds  = torchvision.datasets.MNIST(root=save_path, train=False, download=True)

    # Объединяем train и test в один массив
    X = torch.cat([train_ds.data, test_ds.data], dim=0).numpy()
    X = X[:, np.newaxis, :, :]
    y = torch.cat([train_ds.targets, test_ds.targets], dim=0).numpy()

    # Предобрабатываем данные
    # Нормализуем в [0, 1], поскольку изначально это 0–255
    X = X.astype("float32") / 255.0
    
    # Разделяем на выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state, shuffle=True, stratify=y)
    X_val, y_val = None, None
    if val_size is not None:
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=random_state, shuffle=True, stratify=y_train)

    return X_train, X_val, X_test, y_train, y_val, y_test


def load_steel_dataset(
        path_save: str | Path,
        size_test: float = 0.3,
        size_val: float | None = None,
        window: int = 1
    ) -> tuple[np.ndarray]:
    path_csv = Path(path_save) / 'Steel_industry_data.csv'
    if not os.path.exists(path_csv):
        # Выгрузка архива с данными
        url = "https://www.kaggle.com/api/v1/datasets/download/csafrit2/steel-industry-energy-consumption"
        response = requests.get(url, allow_redirects=True)

        # Сохранение архива
        path_zip = Path(path_save) / path_csv.name.replace('.csv', '.zip')
        with open(path_zip, "wb") as f:
            f.write(response.content)

        # Распаковка
        with zipfile.ZipFile(path_zip, 'r') as zip_ref:
            zip_ref.extractall(path_save)

    # Загрузка данных
    df = pd.read_csv(path_csv)

    # Предобработка данных
    # Переименование столбцов
    df = df.rename(columns={
        'Lagging_Current_Reactive.Power_kVarh' : 'Lagging_Current_Reactive_Power_kVarh',
        'CO2(tCO2)' : 'CO2'
    })
    # Приведение даты к индексу
    df['date'] = pd.to_datetime(df['date'], format="%d/%m/%Y %H:%M")
    df = df.sort_values('date')
    df = df.set_index('date')
    # Преобразование категориальных признаков в числовые
    df = pd.get_dummies(df, columns=['WeekStatus', 'Day_of_week', 'Load_Type'], drop_first=True, dtype=int)

    # Разделяем на выборки
    ind_train = int(len(df)*(1 - size_val - size_test)) if size_val is not None else int(len(df)*(1 - size_test))
    ind_val = ind_train + int(len(df)*size_val) if size_val is not None else ind_train

    # Масштабирование данных
    scaler = MinMaxScaler()
    train_df = df.iloc[:ind_train]
    val_df = df.iloc[ind_train:ind_val] if size_val is not None else None
    test_df  = df.iloc[ind_val:]

    scaler.fit(train_df)
    train_scaled = scaler.transform(train_df)
    val_scaled  = scaler.transform(val_df) if val_df is not None else None
    test_scaled  = scaler.transform(test_df)

    train_scaled = pd.DataFrame(train_scaled, index=train_df.index, columns=df.columns)
    val_scaled  = pd.DataFrame(val_scaled, index=val_df.index, columns=df.columns) if val_scaled is not None else None
    test_scaled  = pd.DataFrame(test_scaled, index=test_df.index, columns=df.columns)

    # Создания временных окон
    X_train, y_train = utils.create_sequences(train_scaled, 'Usage_kWh', window=window, type_task='many_to_many')
    X_val, y_val = utils.create_sequences(val_scaled, 'Usage_kWh', window=window, type_task='many_to_many') if val_scaled is not None else (None, None)
    X_test, y_test = utils.create_sequences(test_scaled,  'Usage_kWh', window=window, type_task='many_to_many')

    return X_train, X_val, X_test, y_train, y_val, y_test

from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from rfpimp import *
from treeinterpreter import treeinterpreter as ti
import numpy as np
import warnings
import shap  # Добавляем импорт shap


def objective_classifier(trial, X_train, y_train, n_trees = 150):
    """
    Целевая функция для оптимизации гиперпараметров Optuna.
    Обучает RandomForestClassifier с предложенными параметрами и возвращает OOB-оценку.

    Args:
        trial (optuna.trial.Trial): Объект Optuna для предложения гиперпараметров.
        X_train (pd.DataFrame): Обучающий набор данных (признаки).
        y_train (pd.Series): Метки обучающего набора данных (целевая переменная).
        n_trees (int): Максимальное количество деревьев для оптимизации.

    Return:
        float: OOB-оценка (Out-of-Bag Score) обученной модели.
    """
    # Определение пространства поиска гиперпараметров
    n_estimators = trial.suggest_int('n_estimators', 25, n_trees)
    max_depth = trial.suggest_int('max_depth', 1, 15, log=True)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 150, 400)
    max_features = trial.suggest_categorical(
        'max_features', ["sqrt", 0.25, 1/3, 0.5, 0.7, 1.0])

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42,
        oob_score=True,
        bootstrap=True,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    return model.oob_score_


def sklearn_importance(X, y, n_trees = 150, shap_sample_size = 500):
    """
    Выполняет оптимизацию гиперпараметров с помощью Optuna, 
    обучает финальную модель с лучшими параметрами и вычисляет важность признаков.

    Args:
        X (pd.DataFrame): Обучающий набор данных (признаки).
        y (pd.Series): Метки обучающего набора данных (целевая переменная).
        n_trees (int): Максимальное количество деревьев для оптимизации.
        shap_sample_size (int): Размер выборки для расчета SHAP значений (для ускорения).

    Return:
        tuple: Кортеж, содержащий:
            1. pd.Series: Важность признаков из sklearn (MDI), отсортированная по убыванию.
            2. pd.Series: Важность признаков из rfpimp (MDA), отсортированная по убыванию.
            3. pd.Series: Важность признаков из treeinterpreter, отсортированная по убыванию.
            4. pd.Series: Важность признаков из SHAP, отсортированная по убыванию.
            5. RandomForestClassifier: Обученная модель с лучшими гиперпараметрами.
            6. optuna.study.Study: Объект исследования Optuna.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    # Создание исследования Optuna для максимизации OOB-оценки
    study = optuna.create_study(direction='maximize')

    # Запуск оптимизации
    study.optimize(lambda trial: objective_classifier(
        trial, X_train, y_train, n_trees), n_trials=50)

    best_params = study.best_params

    # Создание и обучение финальной модели с лучшими параметрами
    rf = RandomForestClassifier(
        **best_params,
        random_state=42,
        oob_score=True,
        bootstrap=True,
        n_jobs=-1
    )

    print("Лучшие параметры:", best_params)
    rf.fit(X_train, y_train)

    # 1. Важность из Sklearn (MDI)
    fi = pd.Series(rf.feature_importances_,
                   index=X_train.columns).sort_values(ascending=False)

    # 2. Важность из rfpimp (MDA)
    imp = importances(rf, X_test, y_test, n_samples=-1)
    imp_series = pd.Series(imp['Importance'].values,
                           index=imp.index).rename_axis(None)
    imp_series = imp_series.sort_values(ascending=False)

    # 3. Важность из treeinterpreter (Вклады признаков)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prediction, bias, contributions = ti.predict(rf, X_test.values)

    # Для многоклассовой классификации: (образцы, признаки, классы)
    mean_abs_contributions = np.mean(np.abs(contributions), axis=(0, 2))
    ti_series = pd.Series(mean_abs_contributions, index=X_train.columns)
    ti_series = ti_series.sort_values(ascending=False)
    
    # --- ИСПРАВЛЕННЫЙ БЛОК SHAP ---
    print("\n[SHAP] Запуск Permutation Explainer...")

    # 1. Фоновые данные (обязательно для борьбы со смещением)
    # Используем медиану или небольшую выборку
    background = shap.maskers.Independent(X_train, max_samples=100)
    
    # 2. Выборка для объяснения
    test_sample = X_test.sample(min(shap_sample_size, len(X_test)), random_state=42)

    # 3. Явное создание Permutation Explainer
    # Мы используем predict_proba, так как это классификация
    explainer = shap.explainers.Permutation(rf.predict_proba, background, feature_names=X.columns)
    
    # 4. Расчет
    # max_evals должен быть нечетным для Permutation (например, 2 * кол-во признаков * 10 + 1)
    # Или просто достаточно большим числом, например, 1001
    shap_values = explainer(test_sample, max_evals=1001)

    # 5. Извлечение значений для положительного класса
    if len(shap_values.shape) == 3:
        actual_shap_values = shap_values.values[:, :, 1]
    else:
        actual_shap_values = shap_values.values

    # 6. Итоговая серия
    shap_series = pd.Series(
        np.abs(actual_shap_values).mean(axis=0), 
        index=X_train.columns
    ).sort_values(ascending=False)
    return fi, imp_series, ti_series, shap_series, rf, study

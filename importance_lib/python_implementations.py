from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def objective_classifier(trial, X_train, y_train):
    """
    Целевая функция для оптимизации гиперпараметров Optuna.
    Обучает RandomForestClassifier с предложенными параметрами и возвращает OOB-оценку.

    Аргументы:
        trial (optuna.trial.Trial): Объект Optuna для предложения гиперпараметров.
        X_train (pd.DataFrame): Обучающий набор данных (признаки).
        y_train (pd.Series): Метки обучающего набора данных (целевая переменная).
        
    Возвращает:
        float: OOB-оценка (Out-of-Bag Score) обученной модели.
    """
    # Определение пространства поиска гиперпараметров
    n_estimators = trial.suggest_int('n_estimators', 50, 2000)
    max_depth = trial.suggest_int('max_depth', 1, 30, log=True)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 40)
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


def sklearn_importance(X_train, y_train):
    """
    Выполняет оптимизацию гиперпараметров с помощью Optuna, 
    обучает финальную модель с лучшими параметрами и вычисляет важность признаков.

    Аргументы:
        X_train (pd.DataFrame): Обучающий набор данных (признаки).
        y_train (pd.Series): Метки обучающего набора данных (целевая переменная).
        
    Возвращает:
        tuple: Кортеж, содержащий:
            1. pd.Series: Важность признаков, отсортированная по убыванию.
            2. RandomForestClassifier: Обученная модель с лучшими гиперпараметрами.
            3. optuna.study.Study: Объект исследования Optuna.
    """
    # Создание исследования Optuna для максимизации OOB-оценки
    study = optuna.create_study(direction='maximize')
    
    # Запуск оптимизации
    study.optimize(lambda trial: objective_classifier(
        trial, X_train, y_train), n_trials=100)
    
    best_params = study.best_params
    
    # Создание и обучение финальной модели с лучшими параметрами
    rf = RandomForestClassifier(
        **best_params,
        random_state=42,
        oob_score=True,
        bootstrap=True,
        n_jobs=-1
    )
    
    print(best_params)
    rf.fit(X_train, y_train)
    
    # Вычисление и форматирование важности признаков
    fi = pd.Series(rf.feature_importances_, index=X_train.columns)
    
    return fi.sort_values(ascending=False), rf, study

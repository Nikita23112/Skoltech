from rpy2.robjects.vectors import StrVector
import pandas as pd
import numpy as np
import re
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
import warnings
import traceback
from rpy2 import robjects


def clean_feature_names(names):
    """
    Преобразует имена столбцов в формат, безопасный для R.
    Заменяет символы '.', '-', ' ' на нижнее подчеркивание '_', так как R
    автоматически преобразует эти символы при работе с формулами.

    Args:
        names (list or pd.Index): Список или индекс имен признаков.

    Return:
        list: Список "очищенных" имен признаков.
    """
    cleaned_names = [name.replace('.', '_').replace(
        '-', '_').replace(' ', '_') for name in names]
    return cleaned_names


def r_randomforest_importance(X, y, n_estimators=100, max_depth=None, min_samples_leaf=1, max_features='sqrt'):
    """
    Вычисляет важность признаков с помощью пакета 'randomForest' в R через rpy2.
    Использует метрику Importance, тип 2 (%IncMSE).

    Args:
        X (pd.DataFrame): Обучающий набор данных (признаки).
        y (pd.Series): Метки обучающего набора данных (целевая переменная).
        n_estimators (int): Количество деревьев ('ntree'). По умолчанию 100.
        max_depth (int, optional): Максимальная глубина дерева. Если None, R использует ограничение maxnodes=NULL.
        min_samples_leaf (int): Минимальное количество выборок в листе ('nodesize'). По умолчанию 1.
        max_features (str or float): Количество признаков для рассмотрения при каждом разбиении ('mtry').
                                     Может быть 'sqrt' или float (доля признаков).
                                     
    Return:
        pd.Series or None: Важность признаков, отсортированная по убыванию, или None в случае ошибки.
    """
    try:
        # Импорт необходимых R-пакетов
        base = importr('base')
        utils = importr('utils')
        randomForest = importr('randomForest') # Пакет Random Forest в R

        original_features = list(X.columns)
        cleaned_features = clean_feature_names(original_features)

        # Подготовка данных для передачи в R
        df_for_r = X.copy()
        df_for_r.columns = cleaned_features
        df_for_r['target'] = y.values

        # Конвертация DataFrame из Pandas (Python) в R DataFrame
        with robjects.conversion.localconverter(robjects.default_converter + pandas2ri.converter) as cv:
            r_df = robjects.conversion.py2rpy(df_for_r)

        # Создание формулы для R (например, target ~ feature_1 + feature_2)
        formula_str = "target ~ " + " + ".join(cleaned_features)
        formula = robjects.Formula(formula_str)

        print(f"Training R Random Forest with formula: {formula_str}")

        n_features = len(cleaned_features)
        # Расчет параметра mtry (количество признаков для случайного выбора)
        if max_features == 'sqrt':
            max_features = round(np.sqrt(n_features))
        else:
            max_features = round(n_features * max_features)

        # Установка ограничения глубины ('maxnodes')
        if max_depth is None:
            maxnodes = robjects.NULL # Эквивалент NULL в R
        else:
            maxnodes=2**max_depth # Преобразование max_depth в maxnodes

        # Обучение модели Random Forest в R
        rf_result = randomForest.randomForest(
            formula,
            data=r_df,
            ntree=n_estimators,         # Количество деревьев
            nodesize=min_samples_leaf,  # Минимальный размер листа
            maxnodes=maxnodes,          # Максимальное количество узлов (для ограничения глубины)
            mtry=max_features,          # Количество признаков для выбора при разбиении
            importance=True             # Запрашиваем расчет важности
        )

        # Получение важности признаков (type=2 соответствует %IncMSE)
        importance_r = randomForest.importance(rf_result, type=2, scale=False)

        # Конвертация результатов в Python объекты
        importance_matrix = np.array(importance_r)
        feature_names_r = list(robjects.r['rownames'](importance_r))

        print(f"Importance matrix shape: {importance_matrix.shape}")
        
        # Извлечение только первого столбца (обычно %IncMSE)
        if importance_matrix.ndim == 2:
            importance_values = importance_matrix[:, 0]
        else:
            importance_values = importance_matrix

        # Создание Pandas Series и восстановление исходных имен признаков
        importance_series = pd.Series(
            importance_values, index=original_features)

        print("Successfully computed variable importance")
        
        return importance_series.sort_values(ascending=False)

    except Exception as e:
        warnings.warn(f"R implementation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def r_ranger_importance_air(X, y, n_estimators=100, max_depth=None, min_samples_leaf=1, max_features='sqrt'):
    """
    Вычисляет важность признаков с помощью пакета 'ranger' в R через rpy2.
    'ranger' является более быстрой реализацией Random Forest, особенно подходящей для классификации.
    Использует метрику 'impurity_corrected' для важности.

    Args:
        X (pd.DataFrame): Обучающий набор данных (признаки).
        y (pd.Series): Метки обучающего набора данных (целевая переменная).
        n_estimators (int): Количество деревьев ('num.trees'). По умолчанию 100.
        max_depth (int, optional): Максимальная глубина дерева ('max.depth').
        min_samples_leaf (int): Минимальное количество выборок в листе ('min.node.size'). По умолчанию 1.
        max_features (str or float): Количество признаков для рассмотрения при каждом разбиении ('mtry').
                                     Может быть 'sqrt', float (доля признаков) или int.
                                     
    Return:
        pd.Series or None: Важность признаков, отсортированная по убыванию, или None в случае ошибки.
    """
    try:
        # Импорт необходимых R-пакетов
        base = importr('base')
        ranger = importr('ranger') # Пакет Ranger в R

        original_features = list(X.columns)

        cleaned_features = clean_feature_names(original_features)

        # Подготовка данных: очистка имен столбцов и добавление целевой переменной
        df_for_r = X.copy()
        df_for_r.columns = cleaned_features
        # Целевую переменную конвертируем в строковый тип для ranger, 
        # чтобы она была интерпретирована как фактор (для классификации)
        df_for_r['target'] = y.astype(str).values

        # Создание словаря для восстановления исходных имен признаков после R-анализа
        name_map = dict(zip(cleaned_features, original_features))

        # Конвертация Pandas DataFrame в R DataFrame
        with robjects.conversion.localconverter(robjects.default_converter + pandas2ri.converter):
            r_df = robjects.conversion.py2rpy(df_for_r)

        # Приведение целевой переменной к типу 'factor' в R (обязательно для классификации)
        r_target = base.factor(r_df.rx2('target'))
        
        # Извлечение только признаков и повторное объединение с 'factor' таргетом
        r_cleaned_features = StrVector(cleaned_features)
        r_df_features_only = r_df.rx(True, r_cleaned_features)
        r_df = base.cbind(r_df_features_only, target=r_target) # target ~ . (формула)

        # Создание формулы для R
        formula_str = "target ~ " + " + ".join(cleaned_features)
        formula = robjects.Formula(formula_str)

        print(f"Training R Ranger Forest with formula: {formula_str}")
        
        # Проверка типа целевой переменной в R (для отладки)
        target_type = robjects.r['class'](r_df.rx2('target'))
        
        n_features = len(cleaned_features)
        # Расчет параметра mtry (количество признаков для случайного выбора)
        if max_features == 'sqrt':
            mtry_val = round(np.sqrt(n_features))
        elif isinstance(max_features, float):
            mtry_val = round(n_features * max_features)
        else:
            mtry_val = max_features

        # Установка ограничения глубины ('max.depth')
        if max_depth is None:
            max_depth = robjects.NULL

        # Обучение модели Ranger в R
        rf_result = ranger.ranger(
            formula,
            data=r_df,
            num_trees=n_estimators,         # Количество деревьев
            min_node_size=min_samples_leaf, # Минимальный размер листа
            mtry=mtry_val,                  # Количество признаков для выбора при разбиении
            importance="impurity_corrected",# Метрика важности: скорректированная нечистота Джини
            max_depth=max_depth,            # Максимальная глубина
            classification=True             # Указание типа задачи (классификация)
        )

        # Извлечение результатов важности признаков
        importance_r = rf_result.rx2('variable.importance')
        importance_values = np.array(importance_r)
        feature_names_r = list(robjects.r['names'](importance_r))

        # Создание Pandas Series и восстановление исходных имен признаков
        importance_series = pd.Series(importance_values, index=feature_names_r)
        importance_series.index = importance_series.index.map(name_map)

        print("Successfully computed variable importance for CLASSIFICATION")

        return importance_series.sort_values(ascending=False)

    except Exception as e:
        warnings.warn(f"Ranger implementation failed: {e}")
        traceback.print_exc()
        return None

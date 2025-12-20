from rpy2.robjects import pandas2ri, Formula, numpy2ri
from rpy2.robjects.vectors import StrVector, IntVector, FactorVector
import pandas as pd
import numpy as np
import re
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import warnings
import traceback
from rpy2 import robjects
import os
from rpy2.robjects.conversion import localconverter
import rpy2.robjects as ro
from rpy2.rinterface_lib.embedded import RRuntimeError
from rpy2.robjects.pandas2ri import converter as pandas2ri_converter


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


def r_randomforestsrc_importance(X, y, n_estimators=1000, max_depth=None, min_samples_leaf=5, max_features='sqrt'):
    """
    Вычисляет важность признаков с помощью пакета 'randomForestSRC' в R через rpy2.
    Использует метрику Importance (по умолчанию - пермутационная важность VIMP).

    Args:
        X (pd.DataFrame): Обучающий набор данных (признаки).
        y (pd.Series): Метки обучающего набора данных (целевая переменная).
        n_estimators (int): Количество деревьев ('ntree'). По умолчанию 1000 (рекомендуется для rfsrc).
        max_depth (int, optional): Максимальная глубина дерева. Если None, R не устанавливает ограничений.
        min_samples_leaf (int): Минимальное количество выборок в листе ('nodesize'). По умолчанию 5 (рекомендуется для rfsrc).
        max_features (str or float): Количество признаков для рассмотрения при каждом разбиении ('mtry').
                                     Может быть 'sqrt' или float (доля признаков).

    Return:
        pd.Series or None: Важность признаков, отсортированная по убыванию, или None в случае ошибки.
    """
    try:
        # --- Импорт необходимых R-пакетов ---
        # randomForestSRC содержит главную функцию rfsrc()
        randomForestSRC = importr('randomForestSRC')
        base = importr('base')

        # Определяем, является ли задача классификацией (для rfsrc)
        # Если количество уникальных значений невелико и тип данных - целочисленный/объект
        is_classification = pd.api.types.is_categorical_dtype(
            y) or y.nunique() <= 10 and y.dtype in ['int64', 'int32', 'object']

        original_features = list(X.columns)
        cleaned_features = clean_feature_names(original_features)

        # --- Подготовка данных для передачи в R ---
        df_for_r = X.copy()
        df_for_r.columns = cleaned_features

        # В rfsrc для классификации целевая переменная должна быть фактором
        if is_classification:
            # Преобразование в фактор R
            df_for_r['target'] = base.as_factor(
                ro.vectors.StrVector(y.astype(str).values))
            print("Detected Classification Task. Target variable converted to R Factor.")
        else:
            # Для регрессии используем численные значения
            df_for_r['target'] = y.values
            print("Detected Regression Task.")

        # Конвертация DataFrame из Pandas (Python) в R DataFrame
        with localconverter(ro.default_converter + pandas2ri_converter) as cv:
            r_df = ro.conversion.py2rpy(df_for_r)

        # Создание формулы для R (например, target ~ feature_1 + feature_2)
        formula_str = "target ~ " + " + ".join(cleaned_features)
        formula = Formula(formula_str)

        print(
            f"Training R Random Forest (randomForestSRC::rfsrc) with formula: {formula_str}")

        n_features = len(cleaned_features)

        # --- Расчет параметра mtry (количество признаков для случайного выбора) ---
        if max_features == 'sqrt':
            mtry_val = round(np.sqrt(n_features))
        elif isinstance(max_features, float):
            mtry_val = round(n_features * max_features)
        else:
            mtry_val = max_features

        # --- Настройка ограничения глубины ('depth') ---
        if max_depth is None:
            # В rfsrc нет прямого параметра для ограничения глубины,
            # кроме ограничения размера узла (nodesize).
            # Мы можем использовать nodesize для косвенного контроля.
            # Оставляем max_depth = None (или используем nodesize)
            max_depth_arg = ro.NULL
        else:
            # Установим max.depth = max_depth
            max_depth_arg = int(max_depth)

        # --- Обучение модели Random Forest в R (rfsrc) ---
        rf_result = randomForestSRC.rfsrc(
            formula,
            data=r_df,
            ntree=n_estimators,         # Количество деревьев
            nodesize=min_samples_leaf,  # Минимальный размер листа
            mtry=mtry_val,              # Количество признаков для выбора при разбиении
            # VIMP (Variable Importance) по умолчанию True и вычисляется OOB
            importance='permute',       # Указываем, что хотим пермутационную важность
            # nsplit=10                 # Опционально: количество случайных точек разбиения для каждого узла
        )

        # --- Получение важности признаков (VIMP) ---
        # В rfsrc важность хранится в объекте результата $importance
        # Получаем компонент 'importance'
        importance_r = rf_result.rx2('importance')

        # Конвертация результатов в Python объекты
        importance_values = np.array(importance_r)

        # Имена признаков в rfsrc находятся в атрибутах вектора важности
        feature_names_r = list(importance_r.names)

        print(f"Importance vector length: {len(importance_values)}")

        # Создание Pandas Series и восстановление исходных имен признаков
        # Используем original_features, так как они были в исходном порядке
        importance_series = pd.Series(
            importance_values,
            index=original_features,
            name='randomForestSRC_VIMP'
        )

        print("Successfully computed variable importance using randomForestSRC (VIMP)")
        #

        return importance_series.sort_values(ascending=False)

    except Exception as e:
        warnings.warn(f"randomForestSRC R implementation failed: {e}")
        traceback.print_exc()
        return None


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
        randomForest = importr('randomForest')  # Пакет Random Forest в R

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
            maxnodes = robjects.NULL  # Эквивалент NULL в R
        else:
            maxnodes = 2**max_depth  # Преобразование max_depth в maxnodes

        # Обучение модели Random Forest в R
        rf_result = randomForest.randomForest(
            formula,
            data=r_df,
            ntree=n_estimators,         # Количество деревьев
            nodesize=min_samples_leaf,  # Минимальный размер листа
            # Максимальное количество узлов (для ограничения глубины)
            maxnodes=maxnodes,
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
        ranger = importr('ranger')  # Пакет Ranger в R

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
        # target ~ . (формула)
        r_df = base.cbind(r_df_features_only, target=r_target)

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
            min_node_size=min_samples_leaf,  # Минимальный размер листа
            mtry=mtry_val,                  # Количество признаков для выбора при разбиении
            # Метрика важности: скорректированная нечистота Джини
            importance="impurity_corrected",
            max_depth=max_depth,            # Максимальная глубина
            # Указание типа задачи (классификация)
            classification=True
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


def r_party_cforest_importance_classification(X, y, n_estimators=100, max_depth=None,
                                              min_samples_leaf=1, max_features='sqrt',
                                              conditional=False):
    """
    Вычисляет условную важность признаков (Conditional Permutation Importance) 
    через party::cforest (R) для задачи КЛАССИФИКАЦИИ.

    Args:
        X (pd.DataFrame/np.array): Признаки.
        y (pd.Series/np.array): Целевая переменная (метки классов).
        n_estimators (int): Количество деревьев (ntree).
        max_depth (int | None): Максимальная глубина деревьев (maxdepth). None означает без ограничения (maxdepth=0 в R).
        min_samples_leaf (int): Минимальное количество образцов в листе (minbucket).
        max_features (str | float): Количество признаков для рассмотрения при расщеплении (mtry).
        conditional (bool): Использовать ли условную перестановку (True) или стандартную (False).
        cores (int): Количество ядер для параллельных вычислений. -1 использует все доступные ядра.

    Return:
        pd.Series: Важность признаков, отсортированная по убыванию, или None в случае ошибки.
    """
    try:
        print(f"Starting party::cforest importance calculation...")
        print(f"X shape: {X.shape}, y shape: {y.shape}")

        # --- Импорт R пакетов ---
        party = importr('party')

        # --- Подготовка и преобразование данных в R-DataFrame ---
        if not isinstance(X, pd.DataFrame):
            X_df = pd.DataFrame(
                X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        else:
            X_df = X.copy()

        y_series = pd.Series(y).astype(str) if not isinstance(
            y, pd.Series) else y.copy().astype(str)

        with localconverter(ro.default_converter + pandas2ri.converter):
            r_X = ro.conversion.py2rpy(X_df)
            # Целевая переменная должна быть ФАКТОРОМ для классификации в R
            r_y = ro.vectors.FactorVector(ro.StrVector(y_series.values))

        # Объединяем X и y в один R-DataFrame
        r_df = ro.r.cbind(r_X, y=r_y)
        formula = Formula('y ~ .')

        # --- 1. Расчет mtry (аналог max_features) ---
        n_features = X_df.shape[1]
        if max_features == 'sqrt':
            mtry_val = max(1, int(np.sqrt(n_features)))
        elif isinstance(max_features, float):
            mtry_val = max(1, int(n_features * max_features))
        else:
            mtry_val = max_features

        # --- 2. Настройки для ctree_control (через cforest_unbiased) ---
        ctrl_args = {
            'ntree': n_estimators,
            'mtry': mtry_val
        }

        # min_samples_leaf (minbucket)
        if min_samples_leaf >= 1:
            ctrl_args['minbucket'] = int(min_samples_leaf)

        # max_depth (maxdepth)
        if max_depth is None:
            ctrl_args['maxdepth'] = 0  # 0 в R = нет ограничения
        elif max_depth >= 1:
            ctrl_args['maxdepth'] = int(max_depth)

        ctrl = party.cforest_unbiased(**ctrl_args)

        # --- 4. Обучение модели с параллелизацией ---
        cf_model = party.cforest(
            formula,
            data=r_df,
            controls=ctrl
        )

        # --- 5. Вычисление важности ---
        print(f"Calculating importance (conditional={conditional})...")
        importance_r = party.varimp(cf_model, conditional=conditional)

        # --- Форматирование результата ---
        importance_values = np.array(importance_r)
        feature_names = list(X_df.columns)

        importance_series = pd.Series(
            importance_values,
            index=feature_names,
            name='party_cforest_importance'
        ).sort_values(ascending=False)

        print(
            f"Success! Importance range: [{importance_series.min():.6f}, {importance_series.max():.6f}]")
        print(f"Top 3 features:\n{importance_series.head(3)}")

        return importance_series

    except Exception as e:
        print(f"Error in party::cforest: {e}")
        import traceback
        traceback.print_exc()
        return None


def r_partykit_importance(X, y, n_estimators=100, max_depth=None,
                          min_samples_leaf=1, max_features='sqrt',
                          conditional=False):
    """
    Расчет важности признаков с использованием пакета R 'partykit' (алгоритм cforest).

    Параметры:
    ----------
    X : array-like или pd.DataFrame
        Матрица признаков. Если передается не DataFrame, колонки будут названы f_0, f_1 и т.д.

    y : array-like
        Вектор целевой переменной. Принудительно преобразуется в фактор (для классификации).

    n_estimators : int, default=100
        Количество деревьев в случайном лесу (параметр ntree в R).

    max_depth : int или None, default=None
        Максимальная глубина деревьев. None (или 0 в R) означает неограниченную глубину.

    min_samples_leaf : int, default=1
        Минимальное количество объектов в терминальном узле (параметр minbucket в R).

    max_features : str, float или int, default='sqrt'
        Количество признаков, случайно выбираемых для каждого разбиения (параметр mtry в R):
        - 'sqrt': квадратный корень из общего числа признаков.
        - float: доля от общего числа признаков (например, 0.5).
        - int: точное количество признаков.

    conditional : bool, default=False
        Тип расчета важности (Permutation Importance):
        - False: Стандартная важность.
        - True: Условная важность (метод Стробла). Позволяет корректно оценивать 
                важность при наличии сильной корреляции между признаками.

    Возвращает:
    -----------
    pd.Series
        Отсортированная важность признаков.
    """
    try:
        # Импортируем движок partykit
        pk = importr('partykit')

        # Подготовка данных: приведение к DataFrame для сохранения имен колонок
        if not isinstance(X, pd.DataFrame):
            X_df = pd.DataFrame(
                X, columns=[f'f_{i}' for i in range(X.shape[1])])
        else:
            X_df = X.copy()

        # Конвертация целевой переменной в формат фактора R (для классификации)
        y_series = pd.Series(y).astype(str)

        with localconverter(ro.default_converter + pandas2ri.converter):
            r_X = ro.conversion.py2rpy(X_df)
            r_y = ro.vectors.FactorVector(ro.StrVector(y_series.values))

        # Сборка данных в R-формат cbind
        r_df = ro.r.cbind(r_X, y=r_y)
        formula = Formula('y ~ .')

        # Логика расчета mtry (аналог max_features в sklearn)
        n_features = X_df.shape[1]
        if max_features == 'sqrt':
            mtry_val = max(1, int(np.sqrt(n_features)))
        elif isinstance(max_features, float):
            mtry_val = max(1, int(n_features * max_features))
        else:
            mtry_val = int(max_features)

        # Параметры контроля обучения
        control_params = {
            'ntree': n_estimators,
            'mtry': mtry_val,
            'minbucket': float(min_samples_leaf),
            'maxdepth': float(max_depth) if max_depth is not None else 0.0
        }

        print(
            f"Training cforest (partykit) with ntree={n_estimators}, mtry={mtry_val}...")

        # Обучение модели cforest
        cf_model = pk.cforest(
            formula,
            data=r_df,
            **control_params
        )

        # Расчет важности (стандартный или условный)
        print(f"Calculating importance (conditional={conditional})...")
        importance_r = pk.varimp(cf_model, conditional=conditional)

        # Маппинг имен признаков из R на их значения важности
        importance_dict = dict(
            zip(list(ro.r.names(importance_r)), np.array(importance_r)))

        # Формирование финального Series с заполнением пропущенных признаков нулями
        importance_series = pd.Series(
            [importance_dict.get(col, 0.0) for col in X_df.columns],
            index=X_df.columns,
            name='party_cforest_importance'
        ).sort_values(ascending=False)

        return importance_series

    except Exception as e:
        print(f"Error in r_partykit_importance: {e}")
        return None

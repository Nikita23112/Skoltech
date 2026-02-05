from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from rfpimp import *
from treeinterpreter import treeinterpreter as ti
import numpy as np
import warnings


def objective_classifier(trial, X_train, y_train):
    """
    Целевая функция для оптимизации гиперпараметров Optuna.
    Обучает RandomForestClassifier с предложенными параметрами и возвращает OOB-оценку.

    Args:
        trial (optuna.trial.Trial): Объект Optuna для предложения гиперпараметров.
        X_train (pd.DataFrame): Обучающий набор данных (признаки).
        y_train (pd.Series): Метки обучающего набора данных (целевая переменная).

    Return:
        float: OOB-оценка (Out-of-Bag Score) обученной модели.
    """
    # Определение пространства поиска гиперпараметров
    n_estimators = trial.suggest_int('n_estimators', 25, 150)
    max_depth = trial.suggest_int('max_depth', 1, 15, log=True)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
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


def sklearn_importance(X, y):
    """
    Выполняет оптимизацию гиперпараметров с помощью Optuna, 
    обучает финальную модель с лучшими параметрами и вычисляет важность признаков.

    Args:
        X (pd.DataFrame): Обучающий набор данных (признаки).
        y (pd.Series): Метки обучающего набора данных (целевая переменная).

    Return:
        tuple: Кортеж, содержащий:
            1. pd.Series: Важность признаков из sklearn, отсортированная по убыванию.
            2. pd.Series: Важность признаков из rfpimp, отсортированная по убыванию.
            3. RandomForestClassifier: Обученная модель с лучшими гиперпараметрами.
            4. optuna.study.Study: Объект исследования Optuna.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    # Создание исследования Optuna для максимизации OOB-оценки
    study = optuna.create_study(direction='maximize')

    # Запуск оптимизации
    study.optimize(lambda trial: objective_classifier(
        trial, X_train, y_train), n_trials=50)

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

    # 1. Важность из Sklearn (MDI)
    fi = pd.Series(rf.feature_importances_,
                   index=X_train.columns).sort_values(ascending=False)

    # 2. Важность из rfpimp (MDA)
    imp = importances(rf, X_test, y_test, n_samples=-1)
    imp_series = pd.Series(imp['Importance'].values,
                           index=imp.index).rename_axis(None)
    imp_series = imp_series.sort_values(ascending=False)

    # 3. Важность из treeinterpreter (Вклады признаков)
    # Используем .values, чтобы избежать конфликта имен признаков
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prediction, bias, contributions = ti.predict(rf, X_test.values)

    # Расчет среднего абсолютного вклада
    # contributions имеет форму (n_samples, n_features, n_classes)
    mean_abs_contributions = np.mean(np.abs(contributions), axis=(0, 2))

    ti_series = pd.Series(mean_abs_contributions, index=X_train.columns)
    ti_series = ti_series.sort_values(ascending=False)

    return fi, imp_series, ti_series, rf, study


def ufi_importance_calculation(model, X, y):
    """
    Реализация Unbiased Feature Importance (UFI/MDI-OOB) по статье Zhou & Hooker.
    Специально для RandomForestClassifier из sklearn.

    Метод вычисляет объективную важность признаков (Unbiased Feature Importance) 
    с использованием OOB (Out-of-Bag) выборок для корректной оценки вклада признаков.
    Основная идея: оценивать важность признаков только на тех объектах, 
    которые не участвовали в обучении конкретного дерева (OOB выборки).

    Args:
        model: обученная модель RandomForestClassifier из sklearn
        X: матрица признаков (pandas DataFrame или numpy array)
        y: вектор целевых меток

    Returns:
        pd.Series: важность признаков, отсортированная по убыванию
    """
    # Определяем количество признаков
    n_features = X.shape[1]
    # Инициализируем массив для суммирования важностей по всем деревьям
    importances_sum = np.zeros(n_features)

    # Преобразуем y в числовой формат, если это категориальные метки
    # factorize преобразует категории в числа: ['A', 'B', 'A'] -> [0, 1, 0]
    if not np.issubdtype(y.dtype, np.number):
        y_labels = pd.factorize(y)[0]  # Преобразуем строковые метки в числа
    else:
        y_labels = y.values  # Если уже числовые, берем как есть

    # Определяем количество уникальных классов для задачи классификации
    n_classes = len(np.unique(y_labels))

    # Проходим по всем деревьям в случайном лесу
    for tree in model.estimators_:
        # Получаем общее количество объектов в данных
        n_samples = X.shape[0]

        # Получаем random_state текущего дерева
        # tree.random_state — это int, поэтому создаем объект RandomState
        rs = np.random.RandomState(tree.random_state)

        # Генерируем индексы бутстрэп-выборки для обучения дерева
        # Это те объекты, которые использовались при обучении дерева
        sample_indices = rs.randint(0, n_samples, n_samples)

        # Вычисляем OOB-индексы: объекты, которые НЕ вошли в обучение этого дерева
        # set(range(n_samples)) - все индексы
        # set(sample_indices) - индексы, использованные при обучении
        oob_indices = np.array(
            list(set(range(n_samples)) - set(sample_indices)))

        # Если нет OOB-объектов (маловероятно, но возможно), пропускаем дерево
        if len(oob_indices) == 0:
            continue

        # Извлекаем OOB данные для текущего дерева
        # Проверяем тип X и безопасно извлекаем значения
        if isinstance(X, pd.DataFrame):
            X_oob = X.values[oob_indices]  # Берем .values для DataFrame
        else:
            X_oob = X[oob_indices]  # Для numpy array берем напрямую

        # Соответствующие OOB метки
        y_oob = y_labels[oob_indices]

        # Получаем внутреннюю структуру дерева из sklearn
        tree_ = tree.tree_

        # Определяем, через какие узлы проходят OOB-объекты
        # decision_path возвращает разреженную матрицу пути по дереву
        node_indicator = tree.decision_path(X_oob)
        # leave_id указывает, в какой лист попадает каждый объект
        leave_id = tree.apply(X_oob)

        # Проходим по всем узлам дерева (кроме листьев)
        for i in range(tree_.node_count):
            # Проверяем, является ли узел внутренним (не листом)
            # У листьев children_left = -1
            if tree_.children_left[i] != -1:
                # Получаем признак, который используется для разделения в этом узле
                feature = tree_.feature[i]
                # Индексы левого и правого дочерних узлов
                left_child = tree_.children_left[i]
                right_child = tree_.children_right[i]

                # Определяем, какие OOB-объекты проходят через текущий узел
                # node_indicator[:, i] - вектор, указывающий, проходит ли объект через узел i
                in_node = node_indicator[:, i].toarray().flatten().astype(bool)

                # Если через узел не проходит ни один OOB-объект, пропускаем
                if not np.any(in_node):
                    continue

                # Определяем, какие объекты попадают в левого и правого ребенка
                in_left = node_indicator[:, left_child].toarray(
                ).flatten().astype(bool)
                in_right = node_indicator[:, right_child].toarray(
                ).flatten().astype(bool)

                # Вспомогательная функция для вычисления "чистоты" узла
                # Используется мера Gini impurity или ее эквивалент
                def get_score(mask):
                    """
                    Вычисляет score для подмножества объектов.
                    Для задачи классификации это квадратичная мера чистоты.

                    Args:
                        mask: булев массив, указывающий, какие объекты учитывать

                    Returns:
                        float: score (чистота) подмножества
                    """
                    if not np.any(mask):
                        return 0  # Если подмножество пустое
                    # Подсчитываем количество объектов каждого класса
                    counts = np.bincount(y_oob[mask], minlength=n_classes)
                    # Вычисляем квадратичную сумму: sum(counts^2) / total
                    return np.sum(counts**2) / np.sum(mask)

                # Вычисляем выигрыш (gain) от разделения по этому признаку
                # Формула из статьи Zhou & Hooker: Unbiased Gini Gain
                # gain = (чистота_левого_потомка + чистота_правого_потомка - чистота_родителя)
                # Нормировка на общее количество OOB-объектов
                gain = (get_score(in_left) + get_score(in_right) -
                        get_score(in_node)) / len(oob_indices)

                # Добавляем выигрыш к общей важности данного признака
                importances_sum[feature] += gain

    # Нормируем важности: делим сумму на количество деревьев в лесу
    # Получаем среднюю важность признака по всем деревьям
    ufi_importances = importances_sum / len(model.estimators_)

    # Преобразуем в pandas Series для удобства, сортируем по убыванию
    # Используем названия столбцов из X как индексы
    return pd.Series(ufi_importances, index=X.columns).sort_values(ascending=False)

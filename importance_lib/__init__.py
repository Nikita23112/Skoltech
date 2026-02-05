# Импорт функций из модуля, содержащего реализацию на Scikit-learn (Python)
from .python_implementations import (
    # Функция для оптимизации гиперпараметров и получения важности признаков (реализация на Python/Scikit-learn)
    sklearn_importance,
    # Целевая функция Optuna для настройки гиперпараметров (часть sklearn_importance)
    objective_classifier,
    # Функция расчета важности признаков для unbiased feature importance (альтернативный метод)
    ufi_importance_calculation
)

# Импорт функций из модуля, содержащего реализации через R (с использованием rpy2)
from .r_implementations import (
    # Функция для получения важности признаков с использованием R-пакета 'randomForest'
    r_randomforest_importance,
    # Вспомогательная функция для очистки имен признаков перед передачей в R
    clean_feature_names,
    # Функция для получения важности признаков с использованием R-пакета 'ranger' (расширенная версия Random Forest)
    r_ranger_importance_air,
    # Функция для получения важности признаков с использованием R-пакета 'party' (cforest - conditional random forest)
    r_party_cforest_importance_classification,
    # Функция для получения важности признаков с использованием R-пакета 'randomForestSRC' (survival, regression, classification)
    r_randomforestsrc_importance,
    # Функция для получения важности признаков с использованием R-пакета 'partykit' (современная версия party)
    r_partykit_importance
)

# Импорт функции из модуля 'pic' (предположительно, модуль для визуализации - от "picture")
from .pic import (
    # Функция для создания или отображения визуализации важности признаков
    picture
)


# Версия текущего пакета - используется для управления версиями и документации
__version__ = "1.0.0"

# Список публичных имен, которые будут экспортированы при использовании 'from package import *'.
# Включает основные функции для использования пользователем.
# Примечание: objective_classifier и clean_feature_names не включены в __all__,
# так как они считаются вспомогательными функциями для внутреннего использования
__all__ = [
    'sklearn_importance',      # Основная Python/Scikit-learn реализация
    'picture',                 # Функция визуализации
    'r_randomforest_importance',           # R-реализация через randomForest
    'r_ranger_importance_air',             # R-реализация через ranger
    # R-реализация через party (cforest)
    'r_party_cforest_importance_classification',
    'r_randomforestsrc_importance',        # R-реализация через randomForestSRC
    'r_partykit_importance',               # R-реализация через partykit
    # Альтернативный метод расчета важности признаков
    'ufi_importance_calculation'
]

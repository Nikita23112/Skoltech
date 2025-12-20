# Импорт функций из модуля, содержащего реализацию на Scikit-learn (Python)
from .python_implementations import (
    # Функция для оптимизации гиперпараметров и получения важности признаков (реализация на Python/Scikit-learn)
    sklearn_importance,
    # Целевая функция Optuna для настройки гиперпараметров (часть sklearn_importance)
    objective_classifier
)

# Импорт функций из модуля, содержащего реализации через R (с использованием rpy2)
from .r_implementations import (
    # Функция для получения важности признаков с использованием R-пакета 'randomForest'
    r_randomforest_importance,
    # Вспомогательная функция для очистки имен признаков перед передачей в R
    clean_feature_names,
    # Функция для получения важности признаков с использованием R-пакета 'ranger'
    r_ranger_importance_air,
    r_party_cforest_importance_classification,
    r_randomforestsrc_importance,
    r_partykit_importance
)

# Импорт функции из модуля 'pic' (название предполагает функцию для визуализации)
from .pic import (
    # Функция для создания или отображения визуализации
    picture
)


# Версия текущего пакета
__version__ = "1.0.0"

# Список публичных имен, которые будут экспортированы при использовании 'from package import *'.
# Включает основные функции для использования пользователем.
__all__ = ['sklearn_importance', 'picture',
           'r_randomforest_importance', 'r_ranger_importance_air', 'r_party_cforest_importance_classification', 'r_randomforestsrc_importance', 'r_partykit_importance']

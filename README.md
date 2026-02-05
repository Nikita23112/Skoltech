#  `feature-importance-comparison`

Утилиты и эксперименты для сравнения методов вычисления **Важности Признаков (Feature Importance - FI)** в моделях **Random Forest** с использованием различных инструментов:
1.  **Scikit-learn** (Python) с оптимизацией гиперпараметров через **Optuna**.
2.  **rfpimp** (Python).
3.  **treeinterpreter** (Python).
4.  **ufi_importance_calculation** (Реализация Unbiased Feature Importance (UFI/MDI-OOB) по статье Zhou & Hooker.
    Специально для RandomForestClassifier из sklearn.)
5.  **R** (через `rpy2`) с пакетами **`randomForest`**, **`ranger`**, **`randomForestSRC`**, **`party`**, **`partykit`**.

---

##  Структура Репозитория

| Путь | Описание |
| :--- | :--- |
| `importance_lib/__init__.py` | Главный файл пакета, определяющий публичный API. |
| `importance_lib/python_implementations.py` | Содержит функции на Python с использованием **`sklearn`**, **`rfpimp`**, **`treeinterpreter`** и **`optuna`** для настройки гиперпараметров (HPO) и расчета FI. |
| `importance_lib/r_implementations.py` | Содержит функции для интеграции с R-пакетами **`randomForest`**, **`ranger`**, **`randomForestSRC`**, **`party`**, **`partykit`** через **`rpy2`**. |
| `importance_lib/pic.py` | Содержит функции для **визуализации** (построения сравнительных графиков) важности признаков. |
| `Random_forest.ipynb` | Блокнот с основными экспериментами и демонстрацией кода. |
| `all_feature_importances_combined.csv` | Файл с сохраненными результатами важности признаков из всех экспериментов. |
| `rent.csv` | Датасет, используемый в экспериментах. |
| `Отчет.pdf` | Файл, в котором указаны проблемы и результаты экспериментов. |

---

##  Установка и Зависимости

Для корректной работы всего функционала необходим **Python 3.x** и **R**.

### Установка R-пакетов (Обязательно для функций R)

Для функций, использующих `rpy2`, необходимо, чтобы в вашей R-среде были установлены пакеты: `randomForest` и `ranger`.

Вы можете установить их в консоли R:

```r
install.packages(c("randomForest", "ranger", "randomForestSRC", "party", "partykit"))
````

---

##  Функции и Примеры использования (API)

Весь функционал доступен для импорта из пакета `importance_lib`.

### 1\) Python/Scikit-learn + Python/rfpimp + Optuna (HPO и FI)

Функция `sklearn_importance` выполняет оптимизацию гиперпараметров (**HPO**) для `RandomForestClassifier` с помощью **Optuna** и возвращает важности признаков (**FI**) из лучшей модели в реализации библиотеки **Scikit-learn** и **rfpimp**.

```python
from importance_lib.python_implementations import sklearn_importance

# X_train: признаки (pd.DataFrame), y_train: целевая переменная (pd.Series)
fi_sklearn, _, _, model, study = sklearn_importance(X_train, y_train)

print("Лучшие параметры:", study.best_params)
print("Топ-5 важных признаков (Scikit-learn):\n", fi_sklearn.head())
```

| Функция | Основное назначение | Возвращает |
| :--- | :--- | :--- |
| `sklearn_importance` | HPO и расчет FI (использует OOB-оценку). | Кортеж: отсортированная FI (`pd.Series`) из реализации **Scikit-learn**,  отсортированная FI (`pd.Series`) из реализации **rfpimp**, отсортированная FI (`pd.Series`) из реализации **treeinterpreter**, обученная модель (`RandomForestClassifier`), объект `Optuna Study`. |

-----

### 2\) R Implementations (сравнение FI)

#### `r_randomforest_importance`

Использует традиционную реализацию **`randomForest`** в R.

```python
from importance_lib.r_implementations import r_randomforest_importance

fi_r_rf = r_randomforest_importance(
    X, y, 
    n_estimators=500, 
    min_samples_leaf=5
)
print("Важность признаков (%IncMSE) из R (randomForest):\n", fi_r_rf.head())
```

#### `r_ranger_importance_air`

Использует быструю реализацию **`ranger`**. Настроена для **классификации** и использует метрику `impurity_corrected`.

```python
from importance_lib.r_implementations import r_ranger_importance_air

# Важно: целевая переменная 'y' будет преобразована в R-фактор
fi_r_ranger = r_ranger_importance_air(
    X, y, 
    n_estimators=500, 
    max_features=0.5
)
print("Важность признаков (impurity_corrected) из R (ranger):\n", fi_r_ranger.head())
```

-----

### 3\) Визуализация

Используйте функцию `picture` для создания сравнительных графиков важности признаков из разных реализаций.

```python
from importance_lib.pic import picture

# Список или словарь с результатами FI
importance_data = [fi_sklearn, fi_r_rf, fi_r_ranger]

picture(
    importance_data, 
    title="Сравнение важности признаков",
    labels=["Scikit-learn", "R (randomForest)", "R (ranger)"]
)
```

-----

##  Примеры и Эксперименты

Пожалуйста, обратитесь к **рабочей тетради `Random_forest.ipynb`** для пошагового примера загрузки данных, запуска всех экспериментов и анализа результатов, описанных в `Отчет.pdf`.

```
```

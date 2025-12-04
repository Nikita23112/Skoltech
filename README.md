-----

````markdown
## `feature-importance-comparison`

Утилиты и эксперименты для сравнения методов вычисления **Важности Признаков (Feature Importance - FI)** в моделях **Random Forest** с использованием различных инструментов:
1.  **Scikit-learn** (Python) с оптимизацией гиперпараметров через **Optuna**.
2.  **R** (через `rpy2`) с пакетами **`randomForest`** и **`ranger`**.

Проект обеспечивает унифицированный интерфейс для получения важности признаков из разных реализаций, что критически важно для надежного анализа данных.

---

## Структура Репозитория

| Путь | Описание |
| :--- | :--- |
| `importance_lib/__init__.py` | Главный файл пакета, определяющий публичный API и управляющий импортом основных функций. |
| `importance_lib/python_implementations.py` | Содержит функции на Python с использованием **`sklearn`** и **`optuna`** для настройки гиперпараметров (HPO) и расчета FI. |
| `importance_lib/r_implementations.py` | Содержит функции для интеграции с R-пакетами **`randomForest`** и **`ranger`** через **`rpy2`**. |
| `importance_lib/pic.py` | Содержит функции для **визуализации** (построения сравнительных графиков) важности признаков. |
| `Random_forest.ipynb` | Блокнот с экспериментами. |
| `all_feature_importances_combined.csv` | Файл в котором сохранены важности признаков во всех экспериментах. |
| `rent.csv` | Датасет, который используется в экспериментах. |
| `Отчет.pdf` | Файл, в котором указаны проблемы и результаты экспериментов. |

---

## Установка и Зависимости

Для корректной работы всего функционала необходим **Python 3.x** и **R**.

### Шаг 2: Установка R-пакетов (Обязательно для функций R)

Для функций, использующих `rpy2`, необходимо, чтобы в вашей R-среде были установлены пакеты: `randomForest` и `ranger`.

Вы можете установить их в консоли R:

```r
install.packages(c("randomForest", "ranger"))
```

-----

## Функции и Примеры использования (API)

Весь функционал доступен для импорта из пакета `feature_importance_pkg`.

### 1\) Python/Scikit-learn + Optuna (HPO и FI)

Функция выполняет оптимизацию гиперпараметров (**HPO**) для `RandomForestClassifier` с помощью **Optuna** и возвращает важность признаков (**FI**) из лучшей модели.

```python
from feature_importance_pkg import sklearn_importance

# Предполагаем, что X_train и y_train — это pd.DataFrame и pd.Series
# X_train: признаки, y_train: целевая переменная
fi_sklearn, model, study = sklearn_importance(X_train, y_train)

print("Лучшие параметры:", study.best_params)
print("Топ-5 важных признаков (Scikit-learn):\n", fi_sklearn.head())
```

| Функция | Основное назначение | Возвращает |
| :--- | :--- | :--- |
| `sklearn_importance` | HPO и расчет FI (использует OOB-оценку). | Кортеж: отсортированная FI (`pd.Series`), обученная модель (`RandomForestClassifier`), объект `Optuna Study`. |

-----

### 2\) R Implementations (сравнение FI)

#### `r_randomforest_importance`

Использует традиционную реализацию **`randomForest`** в R.

```python
from feature_importance_pkg import r_randomforest_importance

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
from feature_importance_pkg import r_ranger_importance_air

# Важно: целевая переменная 'y' будет преобразована в R-фактор
fi_r_ranger = r_ranger_importance_air(
    X, y, 
    n_estimators=500, 
    max_features=0.5
)
print("Важность признаков (impurity_corrected) из R (ranger):\n", fi_r_ranger.head())
```

### 3\) Визуализация

Используйте функцию `picture` для создания сравнительных графиков важности признаков из разных реализаций.

```python
from feature_importance_pkg import picture

# Список или словарь с результатами FI
importance_data = [fi_sklearn, fi_r_rf, fi_r_ranger]

picture(
    importance_data, 
    title="Сравнение важности признаков",
    labels=["Scikit-learn", "R (randomForest)", "R (ranger)"]
)
```

-----

## Примеры и Эксперименты

Пожалуйста, обратитесь к **рабочим тетрадям (`notebooks/`)** или тестовым скриптам в вашем репозитории для пошагового примера загрузки данных и проведения полного эксперимента по сравнению важности признаков.

```
```

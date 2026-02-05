import matplotlib.pyplot as plt
import numpy as np


def picture(importance_list, titles):
    """
    Рисует графики важности признаков, размещая не более 7 графиков в одной строке.

    Args:
        importance_list (list): Список из объектов pd.Series с важностью признаков.
        titles (list): Список названий для каждого графика.
    """
    n_plots = len(importance_list)
    # Максимальное количество столбцов
    ncols = 5
    # Вычисляем необходимое количество строк
    nrows = int(np.ceil(n_plots / ncols))

    # Устанавливаем общий размер рисунка
    # Ширина: 10 дюймов на столбец, Высота: 12 дюймов на строку
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(25, 5 * nrows),  # Оптимальный баланс для 7 колонок
        dpi=100                  # Четкость отрисовки
    )

    # Преобразуем axes в одномерный массив для удобной итерации,
    # даже если это один график (скаляр) или одна строка/столбец (1D массив).
    axes = axes.flatten()

    # Итерируемся по списку важности и рисуем каждый график
    for i, (importance_series, title) in enumerate(zip(importance_list, titles)):
        # Получаем текущую подобласть (Axes) из одномерного массива
        ax = axes[i]

        # Рисуем столбчатую диаграмму
        importance_series.plot(kind='bar', ax=ax)
        ax.set_title(title, fontsize=21)
        ax.set_ylabel('Важность признака', fontsize=15)
        # Поворачиваем метки X для лучшей читаемости
        ax.tick_params(axis='x', which='major', labelsize=14, rotation=90)
        ax.tick_params(axis='y', which='major', labelsize=15)
        ax.grid(axis='y', linestyle='--')

    # Удаляем неиспользуемые подобласти (если общее количество графиков не кратно 7)
    for j in range(n_plots, nrows * ncols):
        fig.delaxes(axes[j])

    # Оптимизируем компоновку, чтобы избежать перекрытия
    plt.tight_layout()
    plt.show()
    return

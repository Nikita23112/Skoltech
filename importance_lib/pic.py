import matplotlib.pyplot as plt


def picture(importance_list, titles):
    """
    Рисует графики важности признаков, размещая не более 3-х графиков в одной строке.

    Args:
        importance_list (list): Список из объектов pd.Series с важностью признаков.
        titles (list): Список названий для каждого графика.
    """
    n_plots = len(importance_list)
    # Максимальное количество столбцов
    ncols = 3 
    # Вычисляем необходимое количество строк
    nrows = int(np.ceil(n_plots / ncols)) 
    
    # Устанавливаем общий размер рисунка
    # Ширина: 5 дюймов на столбец, Высота: 6 дюймов на строку
    fig, axes = plt.subplots(
        nrows=nrows, 
        ncols=ncols, 
        figsize=(5 * ncols, 6 * nrows) 
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
        ax.set_title(title, fontsize=14)
        ax.set_ylabel('Важность признака')
        ax.tick_params(axis='x', rotation=90) # Поворачиваем метки X для лучшей читаемости
        ax.grid(axis='y', linestyle='--')

    # Удаляем неиспользуемые подобласти (если общее количество графиков не кратно 3)
    for j in range(n_plots, nrows * ncols):
        fig.delaxes(axes[j])
        
    # Оптимизируем компоновку, чтобы избежать перекрытия
    plt.tight_layout()
    plt.show()
    return

import matplotlib.pyplot as plt


def picture(fi):
    fi.plot(kind='bar')
    plt.title('feature importances')
    plt.show()
    return

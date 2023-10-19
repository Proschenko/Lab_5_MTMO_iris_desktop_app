import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
dict1 = {'0': "Setosa", '1': "Versicolor", '2': "Virginica"}

def plot_iris_scatter(feature_x, feature_y, xlabel, ylabel):
    # Загрузка датасета ириса
    iris = load_iris()
    #data = iris.data
    target = iris.target


    file_path = "data/iris.csv"
    df = pd.read_csv(file_path, sep=",")

    data=df.iloc[:, :4].copy()
    data=np.array(data)


    # target = df.iloc[:, 4].to_frame().copy()
    # target['Kind'] = pd.factorize(target['Kind'])[0]
    # target=np.array(target)
    # target=target.flatten()






    # Создание графика
    plt.figure(figsize=(8, 6))

    # Построение точек для каждого класса
    for class_value in set(target):
        # Получение индексов точек, принадлежащих текущему классу
        indices = (target == class_value)

        # Извлечение признаков для текущего класса
        x = data[indices, feature_x]  # Первый признак
        y = data[indices, feature_y]  # Второй признак

        # Построение точек с цветом, соответствующим текущему классу
        plt.scatter(x, y, label=f"Iris-{dict1[str(class_value)]}")

    # Добавление легенды
    plt.legend()

    # Добавление подписей осей
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Добавление заголовка
    plt.title("Scatter Plot of Iris Dataset")

    # Отображение графика
    plt.show()


# Пример использования функции
plot_iris_scatter(feature_x=1, feature_y=3, xlabel="Sepal Length (cm)", ylabel="Sepal Width (cm)")
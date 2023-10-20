import math
import numpy as np


class CustomKNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    @staticmethod
    def euclidean_distance(x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    @staticmethod
    def hamming_distance(str1, str2):
        if len(str1) != len(str2):
            raise ValueError("Строки должны быть одинаковой длины")

        distance = 0
        for bit1, bit2 in zip(str1, str2):
            if bit1 != bit2:
                distance += 1

        return distance

    @staticmethod
    def chebyshev_distance(point1, point2):
        if len(point1) != len(point2):
            raise ValueError("Точки должны быть одинаковой размерности")

        max_difference = 0
        for coord1, coord2 in zip(point1, point2):
            difference = abs(coord1 - coord2)
            if difference > max_difference:
                max_difference = difference

        return max_difference

    @staticmethod
    def cosine_distance(vector1, vector2):
        if len(vector1) != len(vector2):
            raise ValueError("Векторы должны быть одинаковой длины")

        dot_product = sum(coord1 * coord2 for coord1, coord2 in zip(vector1, vector2))
        magnitude1 = math.sqrt(sum(coord ** 2 for coord in vector1))
        magnitude2 = math.sqrt(sum(coord ** 2 for coord in vector2))

        if magnitude1 == 0 or magnitude2 == 0:
            return 1  # Избегаем деления на ноль
        else:
            return 1 - (dot_product / (magnitude1 * magnitude2))

    def predict(self, X, metrics):
        y_pred = [self._predict(x, metrics) for x in X]
        return np.array(y_pred)

    def _predict(self, x, metrics):
        match metrics:
            case 1:
                distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
            case 2:
                distances = [self.hamming_distance(x, x_train) for x_train in self.X_train]
            case 3:
                distances = [self.chebyshev_distance(x, x_train) for x_train in self.X_train]
            case 4:
                distances = [self.cosine_distance(x, x_train) for x_train in self.X_train]
            case _:
                distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]

        k_indices = np.argsort(distances)[:self.k]

        k_nearest_labels = [self.y_train[i] for i in k_indices]

        most_common = np.bincount(k_nearest_labels).argmax()
        return most_common

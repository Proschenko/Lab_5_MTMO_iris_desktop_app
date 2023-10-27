import numpy as np


class CustomKNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y, weights):
        self.X_train = X
        self.y_train = y
        self.weights = weights

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum(((x1 - x2) * self.weights) ** 2))

    def manhattan_distance(self, point1, point2):
        if len(point1) != len(point2) or len(point1) != len(self.weights):
            raise ValueError("Точки и веса должны быть одинаковой размерности")

        distance = 0
        for coord1, coord2, weight in zip(point1, point2, self.weights):
            distance += abs(coord1 - coord2) * weight

        return distance

    def chebyshev_distance(self, point1, point2):
        if len(point1) != len(point2):
            raise ValueError("Точки должны быть одинаковой размерности")

        max_difference = 0
        for coord1, coord2, weight in zip(point1, point2, self.weights):
            difference = abs(coord1 - coord2) * weight
            if difference > max_difference:
                max_difference = difference

        return max_difference

    def cosine_distance(self, vector1, vector2):
        if len(vector1) != len(vector2):
            raise ValueError("Векторы должны быть одинаковой длины")

        weighted_vector1 = vector1 * self.weights
        weighted_vector2 = vector2 * self.weights

        dot_product = np.sum(weighted_vector1 * weighted_vector2)
        magnitude1 = np.sqrt(np.sum(weighted_vector1 ** 2))
        magnitude2 = np.sqrt(np.sum(weighted_vector2 ** 2))

        if magnitude1 == 0 or magnitude2 == 0:
            return 1  # Избегаем деления на ноль
        else:
            return 1 - (dot_product / (magnitude1 * magnitude2))

    def predict(self, X, metrics, voices: bool):
        y_pred = [self._predict(x, metrics, voices) for x in X]
        return np.array(y_pred)

    def _predict(self, x, metrics, voices: bool):
        match metrics:
            case 1:
                distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
            case 2:
                distances = [self.manhattan_distance(x, x_train) for x_train in self.X_train]
            case 3:
                distances = [self.chebyshev_distance(x, x_train) for x_train in self.X_train]
            case 4:
                distances = [self.cosine_distance(x, x_train) for x_train in self.X_train]
            case _:
                distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]

        k_distances = np.argsort(distances)[:self.k]

        k_nearest_labels = [self.y_train[i] for i in k_distances]

        if voices:
            dictionary = [(label, 100 / distance) for label, distance in zip(k_nearest_labels, k_distances)]
            # print(dictionary)
            resulgitпшеt_dictionary = {}
            for key, value in dictionary:
                if key in result_dictionary:
                    result_dictionary[key] += value
                else:
                    result_dictionary[key] = value
            # print(result_dictionary)
            most_common = max(result_dictionary, key=result_dictionary.get)

        else:
            most_common = np.bincount(k_nearest_labels).argmax()

        return most_common

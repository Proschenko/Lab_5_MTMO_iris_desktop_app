from CustomKNN import CustomKNN
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X = iris.data[:, :4]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

custom_knn = CustomKNN(3)
custom_knn.fit(X_train, y_train)
y_pred = custom_knn.predict(X_test)
count = 0
for i in range(len(y_pred)):
    if y_pred[i] == y_test[i]:
        count += 1
print(count / len(y_pred) * 100, "%")

dictionary = dict()

dictionary = dict()

for i in range(len(X)):
    for j in range(len(X)):
        if y[i] != y[j]:
            key = (y[i], y[j])
            distance = custom_knn.euclidean_distance(X[i], X[j])
            if key in dictionary:
                if distance < dictionary[key][2]:
                    dictionary[key] = [i, j, distance]
            else:
                dictionary[key] = [i, j, distance]

keys = [(0, 1), (1, 2), (2, 0)]
for key, values in dictionary.items():
    if key in keys:
        print(f"Минимальное расстояние между классами {key}:")
        print(f"Индексы: {values[0]} и {values[1]}")
        print(f"Расстояние: {values[2]}")


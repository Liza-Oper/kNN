import numpy as np
import matplotlib.pyplot as plt

class KNN:
    def __init__(self, k=4):
        self.k = k
        self.X_tr = None
        self.y_tr = None

    def fit(self, X, y):
        self.X_tr = X
        self.y_tr = y

    def pred(self, X):
        preds = []
        for x_test in X:
            D = [np.sqrt(np.sum((x_train - x_test) ** 2)) for x_train in self.X_tr]
            near_n = np.argsort(D)[:self.k]
            knn_labels = [self.y_tr[i] for i in near_n]
            common = max(set(knn_labels), key=knn_labels.count)
            preds.append(common)
        return preds

point = np.array([float(input("X: ")), float(input("Y: "))])

X_train = np.random.rand(50, 2)
y_train = np.random.randint(2, size=50)

print("X_train:", X_train)
print("y_train:", y_train)
print("Точка для классификации:", point)

clf = KNN(k=4)
clf.fit(X_train, y_train)
preds = clf.pred([point])

print("Предсказанный класс для точки:", preds[0])

# Визуализация
plt.figure('KNN')
plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], c='blue', marker='o', label='Класс 0 (Тренировочные данные)')
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], c='orange', marker='o', label='Класс 1 (Тренировочные данные)')
plt.scatter(point[0], point[1], c='blue' if preds[0] == 0 else 'orange', marker='s', label=f'Точка для классификации')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('KNN Классификация')
plt.legend()
plt.show()

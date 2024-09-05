import numpy as np
import matplotlib.pyplot as plt
import csv
import random
from sklearn.cluster import KMeans

# Функция для записи случайных данных в CSV
def writeCSV(filename, rows, cols):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        for _ in range(rows):
            row = [random.uniform(1.0, 10.0) for _ in range(cols)]
            writer.writerow(row)

# Функция для чтения данных из CSV
def readCSV(filename):
    data = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append([float(x) for x in row])
    return np.array(data)

# Генерация случайных данных и запись в CSV
filename = 'data.csv'
writeCSV(filename, 100, 2)  # 100 строк, 2 столбца

# Чтение данных из CSV
data = readCSV(filename)

# Количество кластеров
k = 6

# Инициализация и обучение KMeans
kmeans = KMeans(n_clusters=k, max_iter=100)
kmeans.fit(data)

# Получение центроидов и присвоение кластеров
centroids = kmeans.cluster_centers_
assignments = kmeans.labels_

# Создание фигуры с двумя подграфиками
fig, axs = plt.subplots(1, 2, figsize=(14, 6))  # 1 строка, 2 столбца

# Подграфик 1: данные до кластеризации
axs[0].scatter(data[:, 0], data[:, 1], s=30, color='gray')
axs[0].set_title("Data Before K-Means Clustering")

# Подграфик 2: данные после кластеризации
colors = ['r', 'g', 'b', 'c', 'm', 'y']
for i in range(len(data)):
    axs[1].scatter(data[i][0], data[i][1], color=colors[assignments[i] % len(colors)], s=30)
for i in range(len(centroids)):
    axs[1].scatter(centroids[i][0], centroids[i][1], color=colors[i % len(colors)], s=100, marker='x')
axs[1].set_title("K-Means Clustering")

# Показать оба графика
plt.show()

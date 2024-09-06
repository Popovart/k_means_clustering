import numpy as np
import matplotlib.pyplot as plt
import csv
import random
from sklearn.cluster import KMeans
from matplotlib.colors import ListedColormap

# Funkce pro zápis náhodných dat do CSV
def writeCSV(filename, rows, cols):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        for _ in range(rows):
            row = [random.uniform(1.0, 10.0) for _ in range(cols)]
            writer.writerow(row)

# Funkce pro čtení dat z CSV
def readCSV(filename):
    data = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append([float(x) for x in row])
    return np.array(data)

# Generování náhodných dat a zápis do CSV
filename = 'data.csv'
writeCSV(filename, 100, 2)  # 100 řádků, 2 sloupce

# Čtení dat z CSV
data = readCSV(filename)

# Počet clusterů
k = 5  # Příklad s více než 20 clustery

# Inicializace a trénování KMeans
kmeans = KMeans(n_clusters=k, max_iter=100)
kmeans.fit(data)

# Získání centroidů a přiřazení clusterů
centroids = kmeans.cluster_centers_
assignments = kmeans.labels_

# Vytvoření obrázku s dvěma podgrafy
fig, axs = plt.subplots(1, 2, figsize=(14, 6))  # 1 řádek, 2 sloupce

# Podgraf 1: data před klastrováním
axs[0].scatter(data[:, 0], data[:, 1], s=30, color='gray')
axs[0].set_title("Data před K-Means klastrováním")

# Funkce pro generování požadovaného počtu barev
def generate_colors(num_colors):
    cmap = plt.cm.get_cmap('tab20b', num_colors)  # Použití větší palety
    return cmap(np.linspace(0, 1, num_colors))

# Generování barev pro clustery
colors = generate_colors(k)

# Podgraf 2: data po klastrování
for i in range(len(data)):
    axs[1].scatter(data[i][0], data[i][1], color=colors[assignments[i] % len(colors)], s=30)
for i in range(len(centroids)):
    axs[1].scatter(centroids[i][0], centroids[i][1], color=colors[i % len(colors)], s=100, marker='x')
axs[1].set_title("K-Means klastrování")

# Zobrazit oba grafy
plt.show()

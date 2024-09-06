import numpy as np
import matplotlib.pyplot as plt
import csv
import random
from KMeans import kmeans 
from matplotlib.colors import ListedColormap

def writeCSV(filename, rows, cols):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        for _ in range(rows):
            row = [random.uniform(1.0, 10.0) for _ in range(cols)]
            writer.writerow(row)

def readCSV(filename):
    data = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append([float(x) for x in row])
    return np.array(data)

def generate_colors(num_colors):
    cmap = plt.cm.get_cmap('tab20b', num_colors)
    return cmap(np.linspace(0, 1, num_colors))

# Generate random data and write to CSV
filename = 'data.csv'
writeCSV(filename, 100, 2)

# Read data from CSV
data = readCSV(filename)

# Number of clusters
k = 2

# Using your kmeans function
centroids, assignments = kmeans(data, k) 

# Create a figure with two subplots
fig, axs = plt.subplots(1, 2, figsize=(14, 6))  

# Subplot 1: data before clustering
axs[0].scatter(data[:, 0], data[:, 1], s=30, color='gray')
axs[0].set_title("Data before K-Means clustering")

# Generate colors for clusters
colors = generate_colors(k)

# Subplot 2: data after clustering
for i in range(len(data)):
    axs[1].scatter(data[i][0], data[i][1], color=colors[assignments[i] % len(colors)], s=30)
for i in range(len(centroids)):
    axs[1].scatter(centroids[i][0], centroids[i][1], color=colors[i % len(colors)], s=100, marker='x')
axs[1].set_title("K-Means clustering")

# Show both plots
plt.show()

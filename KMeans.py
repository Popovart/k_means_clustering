import numpy as np
import random

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def initialize_centroids(data, k):
    # Randomly select k points from the data as centroids
    centroids = data[random.sample(range(len(data)), k)]
    return centroids

# Function to assign each point to the nearest centroid
def assign_clusters(data, centroids):
    clusters = []
    for point in data:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        closest_centroid = np.argmin(distances)
        clusters.append(closest_centroid)
    return np.array(clusters)

# Function to recalculate centroids
def update_centroids(data, clusters, k):
    new_centroids = []
    for i in range(k):
        cluster_points = data[clusters == i]
        if len(cluster_points) > 0:
            new_centroid = np.mean(cluster_points, axis=0)
        else:
            # If the cluster is empty, select a random point
            new_centroid = data[random.randint(0, len(data) - 1)]
        new_centroids.append(new_centroid)
    return np.array(new_centroids)

def kmeans(data, k, max_iter=100, tol=1e-4):
    
    centroids = initialize_centroids(data, k)
    
    for iteration in range(max_iter):
        # Assign points to clusters
        clusters = assign_clusters(data, centroids)
        
        # Recalculate centroids
        new_centroids = update_centroids(data, clusters, k)
        
        # Check if centroids have changed (compare with tolerance)
        centroid_shift = np.sum([euclidean_distance(centroids[i], new_centroids[i]) for i in range(k)])
        
        # Update centroids
        centroids = new_centroids
        
        # Stop if centroid shift is less than the threshold
        if centroid_shift < tol:
            break
    
    return centroids, clusters



# Example
# data = np.array([[random.uniform(1, 10), random.uniform(1, 10)] for _ in range(100)])  # Generate random data
# k = 5  
# centroids, clusters = kmeans(data, k)


# print("Cluster centroids:")
# print(centroids)

# print("Cluster assignments for each point:")
# print(clusters)

import random

import numpy as np


def initialize_centroids_forgy(data, k):
    indices = np.random.choice(len(data), k, replace=False)
    return data[indices]


def initialize_centroids_kmeans_pp(data, k):
    indices = [random.choice([i for i in range(len(data))])]
    for _ in range(k-1):
        closest_centroid_index = -1
        closest_distance = float('inf')
        for i, point in enumerate(data):
            if i in indices:
                continue
            for index in indices:
                distance = np.linalg.norm(data[index] - point)
                if distance < closest_distance:
                    closest_distance = distance
                    closest_centroid_index = i
        indices.append(closest_centroid_index)

    return data[indices]


def assign_to_cluster(data, centroids):
    assignments = []
    for point in data:
        closest_centroid_index = -1
        closest_distance = float('inf')
        for i, centroid in enumerate(centroids):
            distance = np.linalg.norm(centroid - point)
            if distance < closest_distance:
                closest_centroid_index = i
                closest_distance = distance
        assignments.append(closest_centroid_index)
    return assignments


def update_centroids(data, assignments):
    centroids = []
    for i in np.unique(assignments):
        assigned_points = data[assignments == i]
        centroid = np.mean(assigned_points, axis=0)
        centroids.append(centroid)
    return np.array(centroids)


def mean_intra_distance(data, assignments, centroids):
    return np.sqrt(np.sum((data - centroids[assignments, :]) ** 2))


def k_means(data, num_centroids, kmeansplusplus=False):
    # centroids initizalization
    if kmeansplusplus:
        centroids = initialize_centroids_kmeans_pp(data, num_centroids)
    else:
        centroids = initialize_centroids_forgy(data, num_centroids)

    assignments = assign_to_cluster(data, centroids)
    for i in range(100):  # max number of iteration = 100
        print(f"Intra distance after {i} iterations: {mean_intra_distance(data, assignments, centroids)}")
        centroids = update_centroids(data, assignments)
        new_assignments = assign_to_cluster(data, centroids)
        if np.all(new_assignments == assignments):  # stop if nothing changed
            break
        else:
            assignments = new_assignments

    return new_assignments, centroids, mean_intra_distance(data, new_assignments, centroids)


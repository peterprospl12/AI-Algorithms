from k_means import k_means
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_iris():
    data = pd.read_csv("data/iris.data", names=["sepal_length", "sepal_width", "petal_length", "petal_width", "class"])
    print(data)
    classes = data["class"].to_numpy()
    features = data.drop("class", axis=1).to_numpy()
    return data, features, classes


def evaluate(clusters, labels):
    for cluster in np.unique(clusters):
        labels_in_cluster = labels[clusters == cluster]
        print(f"Cluster: {cluster}")
        for label_type in np.unique(labels):
            print(f"Num of {label_type}: {np.sum(labels_in_cluster == label_type)}")


def clustering(kmeans_pp):
    data, features, classes = load_iris()
    intra_class_variance = []
    for i in range(100):
        assignments, centroids, error = k_means(features, 3, kmeans_pp)
        evaluate(assignments, classes)
        intra_class_variance.append(error)
    mean = np.mean(intra_class_variance)
    print(f"Mean intra-class variance: {mean}")
    return np.mean(mean)


if __name__ == "__main__":
    kmean_pp_mean = clustering(kmeans_pp=True)
    kmean = clustering(kmeans_pp=False)
    print()
    print("Kmean++ : ", kmean_pp_mean)
    print("Kmean : ", kmean)



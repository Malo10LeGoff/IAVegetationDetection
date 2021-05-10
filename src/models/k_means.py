from sklearn.cluster import KMeans
from typing import List
import numpy as np
import matplotlib.pyplot as plt


def create_and_fit_kmeans(*, X: List[List[float]], nb_clusters: int) -> KMeans:
    """
    Create and fit the model on a single array transformed from the original image with the function
    convert_pixel_to_2D_figure
    Args :
        - X : List whose elements are tuples representing (NDVI,DBR) of a singlep ixel
    """

    return KMeans(n_clusters=nb_clusters, random_state=0).fit(X)


def print_clustering(*, model: KMeans, X: np.array, y_kmeans: List[int]):
    """ Print the Kmeans clustering of the pixels projected on the 2D space (NDVI, NBR) """
    L0 = [X[i][0] for i in range(0, len(X)) if y_kmeans[i] == 0]
    L0b = [X[i][1] for i in range(0, len(X)) if y_kmeans[i] == 0]
    L1 = [X[i][0] for i in range(0, len(X)) if y_kmeans[i] == 1]
    L1b = [X[i][1] for i in range(0, len(X)) if y_kmeans[i] == 1]
    plt.scatter(
        L0,
        L0b,
        s=10,
        c="red",
        label="Cluster 1",
    )

    plt.scatter(
        L1,
        L1b,
        s=10,
        c="blue",
        label="Cluster 2",
    )
    plt.scatter(
        model.cluster_centers_[:, 0],
        model.cluster_centers_[:, 1],
        s=30,
        c="yellow",
        label="Centroids",
    )
    plt.title("Clusters of pixels")
    plt.xlabel("NDVI")
    plt.ylabel("NBR")
    plt.show()
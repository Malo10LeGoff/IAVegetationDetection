from sklearn.cluster import KMeans
from typing import List


def create_and_fit_kmeans(*, X: List[List[float]]) -> KMeans:
    """
    Create and fit the model on a single array transformed from the original image with the function
    convert_pixel_to_2D_figure
    Args :
        - X : List whose elements are tuples representing (NDVI,DBR) of a singlep ixel
    """

    return KMeans(n_clusters=2, random_state=0).fit(X)

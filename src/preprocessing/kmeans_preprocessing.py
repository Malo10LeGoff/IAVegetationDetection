import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.image as mpimg
from PIL import Image
from typing import List
from numpy import asarray
import numpy as np
from preprocessing.utility_functions import compute_NDVI, compute_NBR


def convert_pixel_to_2D_figure(*, image_array: np.array) -> List[List[float]]:
    """
    Function whose goal is to project a (n,m,13) image in a list whose elements are tuples (NDVI, NBR) computed from
    an operation on a single pixel
    Args :
        - image_array : Array of the image
    """
    rows, columns = len(image_array), len(image_array[0])
    new_array = []
    for i in range(0, rows):
        for j in range(0, columns):
            new_array.append(
                [
                    compute_NDVI(pix=image_array[i][j]),
                    compute_NBR(pix=image_array[i][j]),
                ]
            )
    return new_array
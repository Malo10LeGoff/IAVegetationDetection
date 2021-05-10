import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.image as mpimg
from PIL import Image
from numpy import asarray
from typing import List
from time import sleep
import imageio


def load_dataset(*, path_to_dataset: str) -> List[np.array]:
    """
    Load the images as array in memory
    Args :
        - path_to_dataset : path to the data folder where all the images are gathered
    """

    list_of_files = getListOfFiles(dirName=path_to_dataset)
    dataset = []
    for img_name in list_of_files:
        dataset.append(load_image(path_to_image=img_name))
    return dataset


def getListOfFiles(*, dirName: str):
    """ For the given path, get the List of all files in the directory tree """

    listOfFile = os.listdir(dirName)
    allFiles = list()
    for entry in listOfFile:
        fullPath = os.path.join(dirName, entry)
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(dirName=fullPath)
        else:
            allFiles.append(fullPath)
    return allFiles


def load_image(*, path_to_image: str) -> np.array:
    """ From the path, load the image as an array in memory """

    return imageio.imread(path_to_image)


def print_image_from_path(*, path_to_image: str):
    """ Function to print an image given the path """

    img = mpimg.imread(path_to_image)
    imgplot = plt.imshow(img)
    plt.show()


def print_image_from_array(*, image_array: np.array):
    """ Function to print an image given the path """

    img = Image.fromarray(image_array, mode="1")
    img.save("../results/my2.png")
    img.show()


def compute_NDVI(*, pix: List) -> float:
    """
    Computes the NDVI value for one pixel of our image. There are 13 bands in total and we use only
    the infra-red (index 7) and the red (index 3)
    """

    NDVI = (pix[7] - pix[3]) / (pix[7] + pix[3])
    return NDVI


def compute_NBR(*, pix: List):
    """
    Computes the NBR value for one pixel of our image. There are 13 bands in total and we use only
    the infra-red band (index 7) and the SWIR band (index 12)
    """

    NBR = (pix[7] - pix[12]) / (pix[7] + pix[12])
    return NBR

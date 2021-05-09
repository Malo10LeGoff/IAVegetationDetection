from models.k_means import create_and_fit_kmeans
from preprocessing.utility_functions import load_dataset
from preprocessing.kmeans_preprocessing import *

if __name__ == "__main__":

    ### Load the dataset
    dataset = load_dataset(path_to_dataset="../data/")
    print(f"The dimensions of the images are : {dataset[0].shape}")

    ### Transform the images
    array_reformatted = convert_pixel_to_2D_figure(image_array=dataset[10])
    print(f"The dimensions of the reformatted images are : {len(array_reformatted)}")

    ### Create and fit the model
    model = create_and_fit_kmeans(X=array_reformatted)

    ### Transform back the list of results to an array so we can compare with the original image
    new_image = []
    print(f"length row : {int(len(array_reformatted) / 64)}")
    for i in range(0, int(len(array_reformatted) / 64)):
        print(f"chunk of memory : {model.labels_[i*64 : (i+1)*64]}")
        new_image.append(model.labels_[i : i + 64])
    boula = np.array(new_image)
    print(f"Naw array dim : {boula.shape}")
    img = Image.fromarray(boula, mode="1")
    img.save("../results/my2.png")
    img.show()

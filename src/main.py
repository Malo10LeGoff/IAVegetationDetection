from models.k_means import *
from preprocessing.utility_functions import *
from preprocessing.kmeans_preprocessing import *

### HYPER-PARAMETERS
NB_CLUSTERS = 2

if __name__ == "__main__":

    ### Load the dataset
    dataset = load_dataset(path_to_dataset="../data/")
    print(f"The dimensions of the images are : {dataset[0].shape}")

    ### Transform the chosen image
    array_reformatted, index = convert_pixel_to_2D_figure(image_array=dataset[0])
    print(f"The dimensions of the reformatted images are : {len(array_reformatted)}")

    ### Create and fit the model
    model = create_and_fit_kmeans(X=array_reformatted, nb_clusters=NB_CLUSTERS)

    print(f"Number X data point {len(array_reformatted)}")

    ### Print the results
    list_results = model.labels_
    print(f"Results : {list_results}")
    print(f"Nb Results : {len(list_results)}")

    print_clustering(model=model, X=array_reformatted, y_kmeans=list_results)

    ### Transform back the list of results to an array so we can compare with the original image
    new_image = convert_2D_figure_to_array(y_kmeans=list_results, index=index)

    print_image_from_array(image_array=new_image)

import os
from glob import glob
from keras.models import load_model
import numpy as np
from osgeo import gdal
import pandas as pd
from skimage.io import imread
from random import sample

out_data = pd.DataFrame()

os.environ['KMP_DUPLICATE_LIB_OK']='True'

path_to_home = os.path.expanduser("~")
path_to_testing = "~/Karthik/Land_Use_Classification_using_Satellite_Data/test_set"
path_to_testing = path_to_testing.replace("~", path_to_home)
path_to_model = "~/Karthik/Land_Use_Classification_using_Satellite_Data/models/model_transfer_vgg.hdf5"
path_to_model = path_to_model.replace("~", path_to_home)

batch_size = 32
class_indices = {'0': 0, '1': 1, '2': 2,
                 '3': 3, '4': 4, '5': 5,
                 '6': 6, '7': 7, '8': 8,
                 '9': 9}

testing_files = glob(path_to_testing + "/*.tif")


os.environ['KMP_DUPLICATE_LIB_OK']='True'

# CALCULATE MEAN AND STANDARD DEVIATION FOR ALL BANDS
def getMeanStd(path, n_bands=13, n_max=-1):
    """Get mean and standard deviation from images.
    Parameters
    ----------
    path : str
        Path to training images
    n_bands : int
        Number of spectral bands
    n_max : int
        Maximum number of iterations (-1 = all)
    Return
    ----------
    """
    if not os.path.isdir(path):
        print("Error: Directory does not exist.")
        return 0

    mean_array = [[] for _ in range(n_bands)]
    std_array = [[] for _ in range(n_bands)]

    # iterate over the images
    i = 0
    for tif in glob(path + "*/*.*"):
        if (i < n_max) or (n_max == -1):
            ds = gdal.Open(tif)
            for band in range(n_bands):
                mean_array[band].append(
                    np.mean(ds.GetRasterBand(band + 1).ReadAsArray()))
                std_array[band].append(
                    np.std(ds.GetRasterBand(band + 1).ReadAsArray()))
            i += 1
        else:
            break

    # results
    res_mean = [np.mean(mean_array[band]) for band in range(n_bands)]
    res_std = [np.mean(std_array[band]) for band in range(n_bands)]

    # print results table
    print("Band |   Mean   |   Std")
    print("-" * 28)
    for band in range(n_bands):
        print("{band:4d} | {mean:8.3f} | {std:8.3f}".format(
            band=band, mean=res_mean[band], std=res_std[band]))

    return res_mean, res_std

#CALCULATE MEAN AND STD FOR ALL BANDS
train_mean, train_std = getMeanStd(path="train_set/train/", n_bands=13)

#IMAGE LABEL FROM FILE DIRECTORY NAME
def categorical_label_from_full_file_name(files, class_indices):
    from keras.utils import to_categorical
    import os
    # class label without extension
    base_name = [os.path.dirname(i).split("/")[-1] for i in files]
    # label to indices
    image_class = [class_indices[i] for i in base_name]
    # class indices to one-hot-label
    return to_categorical(image_class, num_classes=len(class_indices))

#NORMALIZE MEAN AND STD FOR ALL BANDS
def preprocessing_image_ms(x):
    mean = [1351.790, 1115.080, 1039.618, 942.969, 1195.943, 2002.830, 2302.581, 732.133, 12.087, 1815.571, 1113.341,
            2601.172]
    std = [64.681, 152.555, 186.426, 276.562, 226.984, 355.609, 455.266, 531.110, 98.984, 1.182, 377.014, 301.393,
           502.325]
    # loop over image channels
    for idx, mean_value in enumerate(mean):
        x[..., idx] -= mean_value
        x[..., idx] /= std[idx]
    return x

model = load_model(path_to_model)

for input_path in testing_files:

    image = np.array(imread(input_path), dtype=float)
    image = preprocessing_image_ms(image)
    labels = model.predict(image, verbose = 0)
    print(labels)

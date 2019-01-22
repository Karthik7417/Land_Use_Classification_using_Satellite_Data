from glob import glob
from osgeo import gdal
import numpy as np
import os


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

#DATA AUGMENTATION
def simple_image_generator(files, class_indices, batch_size=32,
                           rotation_range=0, horizontal_flip=False,
                           vertical_flip=False):
    from skimage.io import imread
    from skimage.transform import rotate
    import numpy as np
    from random import sample, choice

    while True:
        # select batch_size number of samples without replacement
        batch_files = sample(files, batch_size)
        # get one_hot_label
        batch_Y = categorical_label_from_full_file_name(batch_files,
                                                        class_indices)
        # array for images
        batch_X = []
        # loop over images of the current batch
        for idx, input_path in enumerate(batch_files):
            image = np.array(imread(input_path), dtype=float)
            image = preprocessing_image_ms(image)
            # process image
            if horizontal_flip:
                # randomly flip image up/down
                if choice([True, False]):
                    image = np.flipud(image)
            if vertical_flip:
                # randomly flip image left/right
                if choice([True, False]):
                    image = np.fliplr(image)
            # rotate image by random angle between
            # -rotation_range <= angle < rotation_range
            if rotation_range is not 0:
                angle = np.random.uniform(low=-abs(rotation_range),
                                          high=abs(rotation_range))
                image = rotate(image, angle, mode='reflect',
                               order=1, preserve_range=True)
            # put all together
            batch_X += [image]
        # convert lists to np.array
        X = np.array(batch_X)
        Y = np.array(batch_Y)

        yield(X, Y)

#CALCULATE MEAN AND STD FOR ALL BANDS
train_mean, train_std = getMeanStd(path="train_set/train/", n_bands=13)
print(train_mean)
print(train_std)

#NORMALIZE MEAN AND STD FOR ALL BANDS
def preprocessing_image_ms(x, mean = train_mean, std = train_std):
    # loop over image channels
    for idx, mean_value in enumerate(mean):
        x[..., idx] -= mean_value
        x[..., idx] /= std[idx]
    return x
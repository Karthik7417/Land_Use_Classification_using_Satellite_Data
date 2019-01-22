import numpy as np
from skimage.io import imread
from keras.models import load_model
import os
from glob import glob
import pandas as pd

os.environ['KMP_DUPLICATE_LIB_OK']='True'

#SET DIRECTORIES
path_to_home = os.path.expanduser("~")
path_to_testing = "~/Karthik/Land_Use_Classification_using_Satellite_Data/test_set"
path_to_testing = path_to_testing.replace("~", path_to_home)
path_to_model = "~/Karthik/Land_Use_Classification_using_Satellite_Data/models/model_transfer_vgg.hdf5"
path_to_model = path_to_model.replace("~", path_to_home)

#LOAD INPUT IMAGES
path_to_images = glob(path_to_testing + "/*.tif")

out_df = pd.DataFrame()
probs_df = pd.DataFrame()

#NORMALIZE MEAN AND STD FOR ALL BANDS
def preprocessing_image(x):
    # MEAN AND STD CALCULATED FROM TRAINING DATA
    mean = [1351.790, 1115.080, 1039.618, 942.969, 1195.943, 2002.830, 2375.471, 2302.581, 732.133, 12.087, 1815.571,
            1113.341, 2601.172]
    std = [64.681, 152.555, 186.426, 276.562, 226.984, 355.609, 455.266, 531.110, 98.984, 1.182, 377.014, 301.393,
           502.325]
    # loop over image channels
    for idx, mean_value in enumerate(mean):
        x[..., idx] -= mean_value
        x[..., idx] /= std[idx]
    return x

labels = []
filepath=[]
probs = []

#LOAD MODEL
model = load_model(path_to_model)

for idx, path_to_image in enumerate(path_to_images[0:100]):
    #READ IMAGE & PREPROCESS
    image = np.array(imread(path_to_image), dtype=float)
    image = preprocessing_image(image)
    image = image.reshape(1, 64, 64, 13)

    #PREDICT LABELS
    image_classified = model.predict(image, batch_size=1024, verbose=0)
    image_classified_label = np.argmax(image_classified, axis=1)
    labels = np.append(labels, image_classified_label)
    image_classified_prob = np.sort(image_classified, axis=1).flatten()[-1]
    filepath = np.append(filepath, os.path.join('./', os.path.relpath(path_to_image)))
    print(idx, ' || path: ', path_to_image,'|| class :', image_classified_label, '\n')
    print(idx, ' || probs: ', image_classified, '\n')
probs = np.asarray(probs)
#WRITE PREDICTIONS TO CSV
out_df = pd.DataFrame({'filepath': filepath, 'labels': labels})
out_df.to_csv('./output_v2.csv')

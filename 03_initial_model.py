import os
from glob import glob
from keras.applications.vgg16 import VGG16 as VGG
from keras.applications.densenet import DenseNet201 as DenseNet
from keras.optimizers import SGD
from keras.layers import GlobalAveragePooling2D, Dense, Input, Conv2D
from keras.models import Model, Sequential, model_from_json
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
from osgeo import gdal

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

#NORMALIZE MEAN AND STD FOR ALL BANDS
def preprocessing_image_ms(x, mean = train_mean, std = train_std):
    # loop over image channels
    for idx, mean_value in enumerate(mean):
        x[..., idx] -= mean_value
        x[..., idx] /= std[idx]
    return x

# variables
path_to_split_datasets = "~/Documents/OneConcern/train_set"
use_vgg = True
batch_size = 64

class_indices = {'0': 0, '1': 1, '2': 2,
                 '3': 3, '4': 4, '5': 5,
                 '6': 6, '7': 7, '8': 8,
                 '9': 9}
num_classes = len(class_indices)

# contruct path
path_to_home = os.path.expanduser("~")
path_to_split_datasets = path_to_split_datasets.replace("~", path_to_home)
path_to_train = os.path.join(path_to_split_datasets, "train")
path_to_validation = os.path.join(path_to_split_datasets, "validation")

# parameters for CNN
input_tensor = Input(shape=(64, 64, 13))
# introduce a additional layer to get from 13 to 3 input channels
input_tensor = Conv2D(3, (1, 1))(input_tensor)
if use_vgg:
    base_model_imagenet = VGG(include_top=False,
                              weights='imagenet',
                              input_shape=(64, 64, 3))
    base_model = VGG(include_top=False,
                     weights=None,
                     input_tensor=input_tensor)
    for i, layer in enumerate(base_model_imagenet.layers):
        # we must skip input layer, which has no weights
        if i == 0:
            continue
        base_model.layers[i+1].set_weights(layer.get_weights())
else:
    base_model_imagenet = DenseNet(include_top=False,
                                   weights='imagenet',
                                   input_shape=(64, 64, 3))
    base_model = DenseNet(include_top=False,
                          weights=None,
                          input_tensor=input_tensor)
    for i, layer in enumerate(base_model_imagenet.layers):
        # we must skip input layer, which has no weights
        if i == 0:
            continue
        base_model.layers[i+1].set_weights(layer.get_weights())

# serialize model to JSON
base_model_json = base_model.to_json()
with open("model_base.json", "w") as json_file:
    json_file.write(base_model_json)
# serialize weights to HDF5
base_model.save_weights("model_base.h5")
print("Base model saved model to disk")

base_model.save('model_base.hdf5')
print("Model saved as hdf5")

# add a global spatial average pooling layer
top_model = base_model.output
top_model = GlobalAveragePooling2D()(top_model)
# or just flatten the layers
# top_model = Flatten()(top_model)

# let's add a fully-connected layer
if use_vgg:
    # only in VGG19 a fully connected nn is added for classfication
    # DenseNet tends to overfitting if using additionally dense layers
    top_model = Dense(2048, activation='relu')(top_model)
    top_model = Dense(2048, activation='relu')(top_model)
# and a logistic layer
predictions = Dense(num_classes, activation='softmax')(top_model)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# print network structure
model.summary()

# serialize model to JSON
model_json = model.to_json()
with open("model_transfer_vgg.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_transfer_vgg.h5")
print("Model Transfer vgg saved model to disk")

model.save('model_transfer_vgg.hdf5')
print("Model transfer vgg saved as hdf5")

# defining ImageDataGenerators
# ... initialization for training
training_files = glob(path_to_train + "/**/*.tif")
train_generator = simple_image_generator(training_files, class_indices,
                                         batch_size=batch_size,
                                         rotation_range=45,
                                         horizontal_flip=True,
                                         vertical_flip=True)

# ... initialization for validation
validation_files = glob(path_to_validation + "/**/*.tif")
validation_generator = simple_image_generator(validation_files, class_indices,
                                              batch_size=batch_size)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional layers
for layer in base_model.layers:
    layer.trainable = False
# set convolution block for reducing 13 to 3 layers trainable
for layer in model.layers[:2]:
        layer.trainable = True

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])

# generate callback to save best model w.r.t val_categorical_accuracy
if use_vgg:
    file_name = "vgg"
else:
    file_name = "dense"
checkpointer = ModelCheckpoint("../data/models/" + file_name +
                               "_ms_transfer_init." +
                               "{epoch:02d}-{val_categorical_accuracy:.3f}." +
                               "hdf5",
                               monitor='val_categorical_accuracy',
                               verbose=1,
                               save_best_only=True,
                               mode='max')
earlystopper = EarlyStopping(monitor='val_categorical_accuracy',
                             patience=10,
                             mode='max',
                             restore_best_weights=True)
history = model.fit_generator(train_generator,
                              steps_per_epoch=1000,
                              epochs=10000,
                              callbacks=[checkpointer, earlystopper],
                              validation_data=validation_generator,
                              validation_steps=500)
initial_epoch = len(history.history['loss'])+1

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
names = []
for i, layer in enumerate(model.layers):
    names.append([i, layer.name, layer.trainable])
print(names)

if use_vgg:
    # we will freaze the first convolutional block and train all
    # remaining blocks, including top layers.
    for layer in model.layers[:2]:
        layer.trainable = True
    for layer in model.layers[2:5]:
        layer.trainable = False
    for layer in model.layers[5:]:
        layer.trainable = True
else:
    for layer in model.layers[:2]:
        layer.trainable = True
    for layer in model.layers[2:8]:
        layer.trainable = False
    for layer in model.layers[8:]:
        layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])

# generate callback to save best model w.r.t val_categorical_accuracy
if use_vgg:
    file_name = "vgg"
else:
    file_name = "dense"
checkpointer = ModelCheckpoint("../data/models/" + file_name +
                               "_ms_transfer_final." +
                               "{epoch:02d}-{val_categorical_accuracy:.3f}" +
                               ".hdf5",
                               monitor='val_categorical_accuracy',
                               verbose=1,
                               save_best_only=True,
                               mode='max')
earlystopper = EarlyStopping(monitor='val_categorical_accuracy',
                             patience=10,
                             mode='max')
model.fit_generator(train_generator,
                    steps_per_epoch=100,
                    epochs=1000,
                    callbacks=[checkpointer, earlystopper],
                    validation_data=validation_generator,
                    validation_steps=500,
                    initial_epoch=initial_epoch)

# load json and create transfer dense model
json_file = open('model_transfer_vgg_v2.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model_transfer_vgg_v2.h5")
print("Model vgg v2 loaded from disk")
model.save('model_vgg_v2.hdf5')
print("Model vgg v2 saved as hdf5")


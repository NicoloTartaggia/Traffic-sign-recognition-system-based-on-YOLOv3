from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import SGD, Adam
from keras import backend as K
import keras
from urllib.request import urlretrieve
import zipfile
import time
import matplotlib.pyplot as plt
import numpy as np

# data folder must be present in filesystem
extract_path = "./data/"
train_data_dir = './data/Training'
validation_data_dir = './data/Testing'
# dimensions of our images.
img_width, img_height = 32, 32
num_classes = 56


def train_data(train_url):
    zip_dir = "./data/BelgiumTSC_Training.zip"

    print("Downloading Belgium TSC Training Dataset\n")
    urlretrieve(train_url, zip_dir)
    zip_ref = zipfile.ZipFile(zip_dir)

    print("Extracting Zip\n")
    zip_ref.extractall(extract_path)
    zip_ref.close()


def test_data(test_url):
    zip_dir = "./data/BelgiumTSC_Testing.zip"

    print("Downloading Belgium TSC Testing Dataset\n")
    urlretrieve(test_url, zip_dir)
    zip_ref = zipfile.ZipFile(zip_dir)

    print("Extracting Zip\n")
    zip_ref.extractall(extract_path)
    zip_ref.close()


def download_data():
    print("Download Datasets")
    start = time.time()
    train_data("http://btsd.ethz.ch/shareddata/BelgiumTSC/BelgiumTSC_Training.zip")
    test_data("http://btsd.ethz.ch/shareddata/BelgiumTSC/BelgiumTSC_Testing.zip")
    end = time.time()
    print("Downloading Datasets 'BelgiumTSC' took ", end - start, 'seconds')


def cnn_model_1():
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

    model = Sequential()
    model.add(Conv2D(2, kernel_size=(5, 5),
                     strides=(2, 2),
                     padding='same',
                     input_shape=input_shape))
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(200))
    model.add(Activation('relu'))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


def cnn_model_2():
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, kernel_size=(3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, kernel_size=(3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    return model


def cnn_model_3():
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

    model = Sequential()

    model.add(Conv2D(16, kernel_size=(3, 3), padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(16, kernel_size=(3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.125))

    model.add(Conv2D(32, kernel_size=(3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(32, kernel_size=(3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.125))

    model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, kernel_size=(3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.125))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    return model


#http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf
#https://www.kaggle.com/tupini07/predicting-mnist-labels-with-a-cnn-in-keras
def cnn_model_lenet():
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

    model = Sequential()

    # Convolutional Layer 1
    model.add(Conv2D(6, kernel_size=(5, 5), input_shape=input_shape, use_bias=True,
                     padding="same"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))

    # Dropout Layer 1
    model.add(Dropout(rate=0.12))

    # Convolutional Layer 2
    model.add(Conv2D(16, kernel_size=(5, 5), use_bias=True,
                     padding="same"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))

    # Convolutional Layer 3
    model.add(Conv2D(35, kernel_size=(5, 5), use_bias=True,
                     padding="same"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))

    # Flatten convolutional result so we can feed data to fully connected layers
    model.add(Flatten())

    # Fully connected 1
    model.add(Dense(120, use_bias=True))
    model.add(Activation('relu'))

    # Dropout Layer 2
    model.add(Dropout(rate=0.5))

    # Fully connected 1
    model.add(Dense(84, use_bias=True))
    model.add(Activation('relu'))

    model.add(Dense(num_classes, use_bias=True))
    model.add(Activation('softmax'))

    model.compile(optimizer=Adam(lr=0.001),
                  loss="categorical_crossentropy",
                  metrics=['accuracy'])

    return model


def setup_data():
    #download_data()

    # this is the augmentation configuration used for training
    """train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True)"""
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=False)

    # this is the augmentation configuration used for testing:
    # only rescaling
    valid_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        #colormode='grayscale',
        class_mode='categorical')

    validation_generator = valid_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        #color_mode='grayscale',
        class_mode='categorical')

    return train_generator, validation_generator


def plot_loss_accuracy(_model_hist, _epochs, _name):
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, _epochs), _model_hist.history["loss"], label="train_loss")
    plt.plot(np.arange(0, _epochs), _model_hist.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, _epochs), _model_hist.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, _epochs), _model_hist.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy Belgian Traffic Sign Dataset\n Model: " + _name)
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig("./figures/" + _name + ".png")

    return None


def train_test_evaluate(_model, _train_generator, _val_generator, _nb_train_samples, _nb_val_samples,
                        _batch_size, _epochs, _model_name):

    _hist = _model.fit_generator(_train_generator,
                                 steps_per_epoch=_nb_train_samples // _batch_size,
                                 epochs=_epochs,
                                 validation_data=_val_generator,
                                 validation_steps=_nb_val_samples // _batch_size)

    _score = _model.evaluate_generator(_val_generator, _nb_val_samples)
    print("Score for model: Test loss: ", _score[0])
    print("Score for model: Test accuracy: ", _score[1])

    _model.summary()
    plot_loss_accuracy(_hist, _epochs, _model_name)
    _model.save("./models/" + _model_name + ".h5")

    return _model, _hist, _score


if __name__ == '__main__':
    nb_train_samples = 5000
    nb_validation_samples = 1500
    epochs = 100
    batch_size = 16

    train_gen, val_gen = setup_data()

    #print("####################################################################################")
    #print("###################################  CNN1  #########################################")
    #print("####################################################################################")
    #model_cnn_1 = cnn_model_1()
    #m1_model, m1_hist, m1_score = train_test_evaluate(model_cnn_1, train_gen, val_gen, nb_train_samples,
    #                                                  nb_validation_samples, batch_size, epochs, "CNN1")

    print("####################################################################################")
    print("###################################  CNN2  #########################################")
    print("####################################################################################")
    model_cnn_2 = cnn_model_2()
    m2_model, m2_hist, m2_score = train_test_evaluate(model_cnn_2, train_gen, val_gen, nb_train_samples,
                                                      nb_validation_samples, batch_size, epochs, "CNN2")

    """print("####################################################################################")
    print("###################################  CNN3  #########################################")
    print("####################################################################################")
    model_cnn_3 = cnn_model_3()
    m3_model, m3_hist, m3_score = train_test_evaluate(model_cnn_3, train_gen, val_gen, nb_train_samples,
                                                      nb_validation_samples, batch_size, epochs, "CNN3")

    print("####################################################################################")
    print("###################################  LeNet  ########################################")
    print("####################################################################################")
    model_lenet = cnn_model_lenet()
    m4_model, m4_hist, m4_score = train_test_evaluate(model_lenet, train_gen, val_gen, nb_train_samples,
                                                      nb_validation_samples, batch_size, epochs, "LeNet")"""

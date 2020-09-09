from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import Adam
from keras import backend as K
import keras
import time
import matplotlib.pyplot as plt
import numpy as np

train_data_dir = './data/Training'
validation_data_dir = './data/Testing'
img_width, img_height = 32, 32
num_classes = 56


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
    # data augmentation configuration for training
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False)

    # this is the augmentation configuration used for testing:
    # only rescaling
    valid_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

    validation_generator = valid_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')
    return train_generator, validation_generator


def plot_loss_accuracy(model_hist, epochs, name):
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, epochs), model_hist.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), model_hist.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, epochs), model_hist.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), model_hist.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy Belgium Traffic Sign Dataset\n Model: " + name)
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig("./figures/" + name + ".png")

    return None


def train_test_evaluate(model, train_generator, val_generator, nb_train_samples, nb_val_samples,
                        batch_size, epochs, model_name):
    hist = model.fit_generator(train_generator,
                               steps_per_epoch=nb_train_samples // batch_size,
                               epochs=epochs,
                               validation_data=val_generator,
                               validation_steps=nb_val_samples // batch_size)

    score = model.evaluate_generator(val_generator, nb_val_samples)
    print("Test loss: ", score[0])
    print("Test accuracy: ", score[1])

    model.summary()
    plot_loss_accuracy(hist, epochs, model_name)
    model.save("./models/" + model_name + ".h5")

    return model, hist, score


if __name__ == '__main__':
    nb_train_samples = 5000
    nb_validation_samples = 1500
    epochs = 100
    batch_size = 16

    train_gen, val_gen = setup_data()

    print("####################################################################################")
    print("###################################  CNN1  #########################################")
    print("####################################################################################")
    model_cnn_1 = cnn_model_1()
    start1 = time.time()
    m1_model, m1_hist, m1_score = train_test_evaluate(model_cnn_1, train_gen, val_gen, nb_train_samples,
                                                      nb_validation_samples, batch_size, epochs, "CNN1")
    time1 = time.time() - start1
    print(time1)

    print("####################################################################################")
    print("###################################  CNN2  #########################################")
    print("####################################################################################")
    model_cnn_2 = cnn_model_2()
    start2 = time.time()
    m2_model, m2_hist, m2_score = train_test_evaluate(model_cnn_2, train_gen, val_gen, nb_train_samples,
                                                      nb_validation_samples, batch_size, epochs, "CNN2")
    time2 = time.time() - start2
    print(time2)

    print("####################################################################################")
    print("###################################  CNN3  #########################################")
    print("####################################################################################")
    model_cnn_3 = cnn_model_3()
    start3 = time.time()
    m3_model, m3_hist, m3_score = train_test_evaluate(model_cnn_3, train_gen, val_gen, nb_train_samples,
                                                      nb_validation_samples, batch_size, epochs, "CNN3")
    time3 = time.time() - start3
    print(time3)
    print("####################################################################################")
    print("###################################  LeNet  ########################################")
    print("####################################################################################")
    model_lenet = cnn_model_lenet()
    start4 = time.time()
    m4_model, m4_hist, m4_score = train_test_evaluate(model_lenet, train_gen, val_gen, nb_train_samples,
                                                      nb_validation_samples, batch_size, epochs, "LeNet")
    time4 = time.time() - start4
    print(time4)

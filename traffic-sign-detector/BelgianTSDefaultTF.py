import os
import skimage.data
import skimage.transform
import numpy as np
import tensorflow as tf


#
#    Load Data
#


def load_data(data_dir):
    """Loads a data set and returns two lists:

    images: a list of Numpy arrays, each representing an image.
    labels: a list of numbers that represent the images labels.
    """
    labels = []
    images = []
    # Get all subdirectories of data_dir. Each represents a label.
    directories = [d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d))]

    # Loop through the label directories and collect the data in
    # two lists, labels and images.
    for d in directories:
        # print(d)
        label_dir = os.path.join(data_dir, d)
        file_names = [os.path.join(label_dir, f)
                      for f in os.listdir(label_dir) if f.endswith(".ppm")]
        # For each label, load it's images and add them to the images list.
        # And add the label number (i.e. directory name) to the labels list.
        for f in file_names:
            # print(f)
            images.append(skimage.data.imread(f))
            labels.append(int(d))
    print("Unique Labels: {0}\nTotal Images: {1}".format(len(set(labels)), len(images)))
    return images, labels


# Load training and testing datasets.
ROOT_PATH = "./data/"
train_data_dir = os.path.join(ROOT_PATH, "Training")
test_data_dir = os.path.join(ROOT_PATH, "Testing")

# Load Data
images, labels = load_data(train_data_dir)

#
#	Transform images to 32x32 pixels
#
# Resize images
images32 = [skimage.transform.resize(image, (32, 32), mode='constant')
            for image in images]
# Check resized images
for image in images32[:50]:
    print("shape: {0}, min: {1}, max: {2}".format(image.shape, image.min(), image.max()))
# Model training
#


labels_a = np.array(labels)
images_a = np.array(images32)
print("labels: ", labels_a.shape, "\nimages: ", images_a.shape)

# Create a graph to hold the model.
graph = tf.Graph()

# Create model in the graph.
with graph.as_default():
    # Placeholders for inputs and labels.
    images_ph = tf.placeholder(tf.float32, [None, 32, 32, 3])
    labels_ph = tf.placeholder(tf.int32, [None])
    # Flatten input from: [None, height, width, channels]
    # To: [None, height * width * channels] == [None, 3072]
    images_flat = tf.contrib.layers.flatten(images_ph)
    # Fully connected layer.
    # Generates logits of size [None, 62]
    logits = tf.contrib.layers.fully_connected(images_flat, 62, tf.nn.relu)
    # Convert logits to label indexes (int).
    # Shape [None], which is a 1D vector of length == batch_size.
    predicted_labels = tf.argmax(logits, 1)
    # Define the loss function.
    # Cross-entropy is a good choice for classification.
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels_ph))
    # Create training op.
    train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
    # And, finally, an initialization op to execute before training.
    init = tf.global_variables_initializer()
# Create a session to run the graph we created.
session = tf.Session(graph=graph)
# First step is always to initialize all variables.
# We don't care about the return value, though. It's None.
_ = session.run([init])
for i in range(401):
    _, loss_value = session.run([train, loss],
                                feed_dict={images_ph: images_a, labels_ph: labels_a})
    if i % 10 == 0:
        print("Loss: ", loss_value)
#
#	Predictions
#
# Load the test dataset.
test_images, test_labels = load_data(test_data_dir)
# Transform the images, just like we did with the training set.
test_images32 = [skimage.transform.resize(image, (32, 32), mode='constant')
                 for image in test_images]
# Run predictions against the full test set.
predicted = session.run([predicted_labels],
                        feed_dict={images_ph: test_images32})[0]
# Calculate how many matches we got.
match_count = sum([int(y == y_) for y, y_ in zip(test_labels, predicted)])
accuracy = match_count / len(test_labels)
print("Accuracy: {:.3f}".format(accuracy))
# Close the session. This will destroy the trained model.
session.close()

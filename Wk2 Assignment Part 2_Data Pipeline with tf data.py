# ### Part 2: tf.data

# #### The CIFAR-100 Dataset

# In the second part of this assignment, you will use the [CIFAR-100 dataset](
# https://www.cs.toronto.edu/~kriz/cifar.html). This image dataset has 100 classes with 500 training images and 100
# test images per class. * A. Krizhevsky. "Learning Multiple Layers of Features from Tiny Images". April 2009 Your
# goal is to use the tf.data module preprocessing tools to construct a data ingestion pipeline including filtering
# and function mapping over the dataset to train a neural network to classify the images.

# #### Load the dataset

import tensorflow as tf
from tensorflow.keras.datasets import cifar100
import numpy as np
import matplotlib.pyplot as plt
import json
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, Flatten, Dense

# Load the data, along with the labels

(train_data, train_labels), (test_data, test_labels) = cifar100.load_data(label_mode='fine')
with open('Customising your Models TensorFlow 2/Data/cifar100_labels.json', 'r') as j:
    cifar_labels = json.load(j)

# #### Display sample images and labels from the training set

# Display a few images and labels

plt.figure(figsize=(15, 8))
inx = np.random.choice(train_data.shape[0], 32, replace=False)
for n, i in enumerate(inx):
    ax = plt.subplot(4, 8, n + 1)
    plt.imshow(train_data[i])
    plt.title(cifar_labels[int(train_labels[i])])
    plt.axis('off')


# #### Create Dataset objects for the train and test images
#
# You should now write a function to create a `tf.data.Dataset` object for each of the training and test images and
# labels. This function should take a numpy array of images in the first argument and a numpy array of labels in the
# second argument, and create a `Dataset` object.
#
# Your function should then return the `Dataset` object. Do not batch or shuffle the `Dataset` (this will be done
# later).


def create_dataset(data, labels):
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    return dataset


# Run the below cell to convert the training and test data and labels into datasets

train_dataset = create_dataset(train_data, train_labels)
test_dataset = create_dataset(test_data, test_labels)

# Check the element_spec of your datasets

print(train_dataset.element_spec)
print(test_dataset.element_spec)


# #### Filter the Dataset
#
# Write a function to filter the train and test datasets so that they only generate images that belong to a specified
# set of classes.
# The function should take a `Dataset` object in the first argument, and a list of integer class indices in the
# second argument.
#
# Inside your function you should define an auxiliary function that you will use with the `filter` method of the
# `Dataset` object. This auxiliary function should take image and label arguments (as in the `element_spec`)
# for a single element in the batch, and return a boolean indicating if the label is one of the allowed classes.
#
# Your function should then return the filtered dataset.
# **Hint:** you may need to use the [`tf.equal`](https://www.tensorflow.org/api_docs/python/tf/math/equal),
# [`tf.cast`](https://www.tensorflow.org/api_docs/python/tf/dtypes/cast) and
#
# [`tf.math.reduce_any`](https://www.tensorflow.org/api_docs/python/tf/math/reduce_any)
# NOTE This reduce_any uses the logical OR operator ( || ) returns the boolean value true if either or
# both operands is true and returns false otherwise
#
# functions in your auxiliary function.

# Complete the following function.
# Make sure to not change the function name or arguments.

def filter_classes(dataset, classes):
    def filter_auxilary(image, label):
        return tf.math.reduce_any(tf.equal(label, classes))

    return dataset.filter(filter_auxilary)


# Run the below cell to filter the datasets using your function

cifar_classes = [0, 29, 99]  # Your datasets should contain only classes in this list

train_dataset = filter_classes(train_dataset, cifar_classes)
test_dataset = filter_classes(test_dataset, cifar_classes)

# #### Apply map functions to the Dataset
#
# You should now write two functions that use the `map` method to process the images and labels in the filtered dataset.
#
# The first function should one-hot encode the remaining labels so that we can train the network using a
# categorical cross entropy loss.
#
# The function should take a `Dataset` object as an argument. Inside your function you should define an
# auxiliary function that you will use with the `map` method of the `Dataset` object.
# This auxiliary function should take image and label arguments (as in the `element_spec`) for a single
# element in the batch, and return a tuple of two elements, with the unmodified image in the first element,
# and a one-hot vector in the second element. The labels should be encoded according to the following:
#
# * Class 0 maps to `[1., 0., 0.]`
# * Class 29 maps to `[0., 1., 0.]`
# * Class 99 maps to `[0., 0., 1.]`
#
# Your function should then return the mapped dataset.

# Complete the following function.
# Make sure to not change the function name or arguments.
"""
This function should map over the dataset to convert the label to a
one-hot vector. The encoding should be done according to the above specification.
The function should then return the mapped Dataset object.
"""


def map_labels(dataset):
    def map_auxilary(image, label):
        if label == 0:
            return image, tf.constant([1., 0., 0.])
        elif label == 29:
            return image, tf.constant([0., 1., 0.])
        else:
            return image, tf.constant([0., 0., 1.])

    return dataset.map(map_auxilary)


'''
I tried hard to make this work but since tf.one_hot does not accept the categories themselves, 
but instead accepts a list of indices for the One Hot Encoded features, I couldn't figure it out without creating
a "for" loop to replace 0, 29, and 99 with 0,1, and 2. So I figured I might as well use the for loop to create
the one-hot encodings themselves. I really wish I could have gotten this to work though

def map_labels(dataset):
    def to_categorical(image, label):
        label = [tf.one_hot(label,depth=3)]# shape was (1,1,3) so reshape so that it is (3,)
        #label = tf.reshape(label, [3])
        return (image, label)
    return dataset.map(to_categorical)
    '''
# Run the below cell to one-hot encode the training and test labels.
# look at 3 individual elements before one_hot transformation
for elem in train_dataset.take(3):
    print(elem[1])

train_dataset = map_labels(train_dataset)
test_dataset = map_labels(test_dataset)

print(train_dataset.element_spec)
print(test_dataset.element_spec)
# now look after one_hot transformation
for elem in train_dataset.take(3):
    print(elem[1])


# The second function should process the images according to the following specification:
#
# * Rescale the image pixel values by a factor of 1/255.
# * Convert the colour images (3 channels) to black and white images (single channel) by computing the average pixel
# value across all channels.
#
# The function should take a `Dataset` object as an argument. Inside your function you should again define an
# auxiliary function that you will use with the `map` method of the `Dataset` object. This auxiliary function
# should take image and label arguments (as in the `element_spec`) for a single element in the batch, and
# return a tuple of two elements, with the processed image in the first element, and the unmodified label
# in the second argument. Your function should then return the mapped dataset.
#
# **Hint:** you may find it useful to use
# [`tf.reduce_mean`](https://www.tensorflow.org/api_docs/python/tf/math/reduce_mean?version=stable)
# since the black and white image is the colour-average of the colour images.
# You can also use the `keepdims` keyword in `tf.reduce_mean` to retain the single colour channel.

def map_images(dataset):
    def map_auxilary(image, label):
        image = image / 255
        # Reduce axis 2 by mean (= color)
        # i.e. image = [[[r,g,b], ...]]
        # out = [[[ grayvalue ], ... ]] where grayvalue = mean(r, g, b)
        image = tf.reduce_mean(image, axis=2, keepdims=True)
        return image, label

    return dataset.map(map_auxilary)


# Run the below cell to apply your mapping function to the datasets

train_dataset_bw = map_images(train_dataset)
test_dataset_bw = map_images(test_dataset)

train_dataset_bw.element_spec
test_dataset_bw.element_spec
# #### Display a batch of processed images

# Run this cell to view a selection of images before and after processing

plt.figure(figsize=(16, 5))
plt.suptitle("Unprocessed images", fontsize=16)
for n, elem in enumerate(train_dataset.take(10)):
    images, labels = elem
    ax = plt.subplot(2, 5, n + 1)
    plt.title(cifar_labels[cifar_classes[np.where(labels == 1.)[0][0]]])
    plt.imshow(np.squeeze(images), cmap='gray')
    plt.axis('off')

plt.figure(figsize=(16, 5))
plt.suptitle("Processed images", fontsize=16)
for n, elem in enumerate(train_dataset_bw.take(10)):
    images_bw, labels_bw = elem
    ax = plt.subplot(2, 5, n + 1)
    plt.title(cifar_labels[cifar_classes[np.where(labels_bw == 1.)[0][0]]])
    plt.imshow(np.squeeze(images_bw), cmap='gray')
    plt.axis('off')

# We will now batch and shuffle the Dataset objects.

# Run the below cell to batch the training dataset and expand the final dimensinos
# the way this works is the buffer will stay filled with 100 data examples, and the batch of 10 will
# be sampled from the buffer
train_dataset_bw = train_dataset_bw.batch(10)
train_dataset_bw = train_dataset_bw.shuffle(100)

test_dataset_bw = test_dataset_bw.batch(10)
test_dataset_bw = test_dataset_bw.shuffle(100)

# #### Train a neural network model
#
# Now we will train a model using the `Dataset` objects. We will use the model specification and function from the
# first part of this assignment, only modifying the size of the input images.

# Build and compile a new model with our original spec, using the new image size
'''copied and pasted from wk 2 Assignment Part 1'''


def get_model(input_shape):
    inputs = Input(shape=input_shape, name='input_layer 1')
    h = Conv2D(8, (8, 8), activation='relu', name='Conv2D_layer2', padding='same')(inputs)
    h = MaxPooling2D((2, 2), name='maxpool2d_layer3')(h)
    h = Conv2D(4, (4, 4), activation='relu', name='Conv2D_layer4', padding='same')(h)
    h = MaxPooling2D((2, 2), name='maxpool2d_layer5')(h)
    h = Flatten(name='flatten_layer6')(h)
    h = Dense(16, activation='relu', name='Dense_layer7')(h)
    outputs = Dense(3, activation='softmax', name='Out_Dense_Softmax_layer8')(h)
    model = Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


cifar_model = get_model((32, 32, 1))

# Train the model for 15 epochs

history = cifar_model.fit(train_dataset_bw, validation_data=test_dataset_bw, epochs=15)

# #### Plot the learning curves

# Run this cell to plot accuracy vs epoch and loss vs epoch

plt.figure(figsize=(15, 5))
plt.subplot(121)
try:
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
except KeyError:
    try:
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
    except KeyError:
        plt.plot(history.history['categorical_accuracy'])
        plt.plot(history.history['val_categorical_accuracy'])
plt.title('Accuracy vs. epochs')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='lower right')

plt.subplot(122)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss vs. epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper right')
plt.show()

# Create an iterable from the batched test dataset
# (that means we can easily access each element in the dataset by writing a simple for loop)

test_dataset = test_dataset.batch(10)
iter_test_dataset = iter(test_dataset)

# Display model predictions for a sample of test images

plt.figure(figsize=(15, 8))
inx = np.random.choice(test_data.shape[0], 18, replace=False)
images, labels = next(iter_test_dataset)
probs = cifar_model(tf.reduce_mean(tf.cast(images, tf.float32), axis=-1, keepdims=True) / 255.)
preds = np.argmax(probs, axis=1)
for n in range(10):
    ax = plt.subplot(2, 5, n + 1)
    plt.imshow(images[n])
    plt.title(cifar_labels[cifar_classes[np.where(labels[n].numpy() == 1.0)[0][0]]])
    plt.text(0, 35, "Model prediction: {}".format(cifar_labels[cifar_classes[preds[n]]]))
    plt.axis('off')

# ## From the W2 Programming Assignment: Data Pipeline with Keras and TF Data
# https://www.coursera.org/learn/customising-models-tensorflow2/programming/3hWzU/data-pipeline-with-keras-and-tf-data
# Data pipeline with Keras and tf.data

# ### Instructions
# 
# In this notebook, I will implement a data processing pipeline using tools from both Keras and the tf.data module.
# Using the `ImageDataGenerator` class in the tf.keras module I will feed a network with training and test images
# from a local directory containing a subset of the LSUN dataset, and train the model both with and without data augmentation.
# I will then use the `map` and `filter` functions of the `Dataset` class with the CIFAR-100 dataset to train a
# network to classify a processed subset of the images.

import tensorflow as tf
from tensorflow.keras.datasets import cifar100
import numpy as np
import matplotlib.pyplot as plt
import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping

# ### Part 1: tf.keras
# #### The LSUN Dataset
'''The dataset is available for download as a zip file at the following link:
https://drive.google.com/open?id=1T4jFYYLHSuG5zjo1dF3btCLzgS2R2AU-'''

# In the first part of this assignment, I will use a subset of the [LSUN dataset](https://www.yf.io/p/lsun).
# This is a large-scale image dataset with 10 scene and 20 object categories. A subset of the LSUN dataset has been
# provided, and has already been split into training and test sets. The three classes included in the subset are
# `church_outdoor`, `classroom` and `conference_room`.
# 
# * F. Yu, A. Seff, Y. Zhang, S. Song, T. Funkhouser and J. Xia. "LSUN: Construction of a Large-scale Image Dataset
# using Deep Learning with Humans in the Loop". arXiv:1506.03365, 10 Jun 2015
# 
# The goal is to use the Keras preprocessing tools to construct a data ingestion and augmentation pipeline to train
# a neural network to classify the images into the three classes.

# Save the directory locations for the training, validation and test sets

train_dir = 'Customising your Models TensorFlow 2/Data/lsun/lsun/train'
valid_dir = 'Customising your Models TensorFlow 2/Data/lsun/lsun/valid'
test_dir = 'Customising your Models TensorFlow 2/Data/lsun/lsun/test'

# #### Create a data generator using the ImageDataGenerator class

# Write a function that creates an `ImageDataGenerator` object,
# which rescales the image pixel values by a factor of 1/255.


def get_ImageDataGenerator():
    return ImageDataGenerator(rescale=(1/255))

# Call the function to get an ImageDataGenerator as specified

image_gen = get_ImageDataGenerator()


# Now write a function that returns a generator object that will yield batches of images and labels
# from the training and test set directories. The generators should:
# 
# * Generate batches of size 20.
# * Resize the images to 64 x 64 x 3.
# * Return one-hot vectors for labels. These should be encoded as follows:
#     * `classroom` $\rightarrow$ `[1., 0., 0.]`
#     * `conference_room` $\rightarrow$ `[0., 1., 0.]`
#     * `church_outdoor` $\rightarrow$ `[0., 0., 1.]`
# * Pass in an optional random `seed` for shuffling (this should be passed into the `flow_from_directory` method).
# 
# **Hint:** you may need to refer to the
# [documentation](https://keras.io/preprocessing/image/#imagedatagenerator-class) for the `ImageDataGenerator`.

'''The ImageDataGenerator class has three 
methods flow(), 
flow_from_directory() 
and flow_from_dataframe() 
to read the images from a big numpy array and folders containing images.

Yield vs. Return
- Yield returns a generator object to the caller, and the execution of the code starts only when the generator is iterated.	
- A return in a function is the end of the function execution, and a single value is given back to the caller.
I don't have to use "yield" here since I'm not coding the actual generator (already in keras), I'm just creating a 
function that calls the generator'''

def get_generator(image_data_generator, directory, seed=None):
    return image_data_generator.flow_from_directory(directory, target_size=(64, 64),
                                                    classes=['classroom', 'conference_room', 'church_outdoor'],
                                                    class_mode="categorical",
                                                    batch_size=20, seed=seed)

# Run this cell to define training and validation generators

train_generator = get_generator(image_gen, train_dir)
valid_generator = get_generator(image_gen, valid_dir)

# We are using a small subset of the dataset for demonstrative purposes in this assignment.

# #### Display sample images and labels from the training set

# Display a few images and labels from the training set

batch = next(train_generator)
batch_images = np.array(batch[0])
batch_labels = np.array(batch[1])
lsun_classes = ['classroom', 'conference_room', 'church_outdoor']

plt.figure(figsize=(16,10))
for i in range(20):
    ax = plt.subplot(4, 5, i+1)
    plt.imshow(batch_images[i])
    plt.title(lsun_classes[np.where(batch_labels[i] == 1.)[0][0]])
    plt.axis('off')

# Reset the training generator

train_generator = get_generator(image_gen, train_dir)

'''
Build the neural network model
Build and compile a convolutional neural network classifier. 
Using the functional API, build the model according to the following specifications:

The model should use the input_shape in the function argument to define the Input layer.
The first hidden layer should be a Conv2D layer with 8 filters, a 8x8 kernel size.
The second hidden layer should be a MaxPooling2D layer with a 2x2 pooling window size.
The third hidden layer should be a Conv2D layer with 4 filters, a 4x4 kernel size.
The fourth hidden layer should be a MaxPooling2D layer with a 2x2 pooling window size.
This should be followed by a Flatten layer, and then a Dense layer with 16 units and ReLU activation.
The final layer should be a Dense layer with 3 units and softmax activation.
All Conv2D layers should use "SAME" padding and a ReLU activation function.
In total, the network should have 8 layers. 

The model should then be compiled with the Adam optimizer with 
learning rate 0.0005, categorical cross entropy loss, and categorical accuracy metric.'''


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


# Build and compile the model, print the model summary

lsun_model = get_model((64, 64, 3))
lsun_model.summary()

# #### Train the neural network model
# 
# Write a function to train the model for a specified number of epochs (specified in the `epochs` argument).
# The function takes a `model` argument, as well as `train_gen` and `valid_gen` arguments
# for the training and validation generators respectively, which should be used for training and
# validation data in the training run. Incorporate the following callbacks:
# 
# * An `EarlyStopping` callback that monitors the validation accuracy and has patience set to 10. 
# * A `ReduceLROnPlateau` callback that monitors the validation loss and has the factor set to 0.5 and minimum learning set to 0.0001
# 
# Your function should return the training history.

def train_model(model, train_gen, valid_gen, epochs):
    earlystopping = EarlyStopping(patience=10)
    learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.5, min_lr=0.0001)
    training_hx= model.fit(train_gen, validation_data=valid_gen,epochs=epochs,
                                        callbacks=[earlystopping,learning_rate_reduction])
    return training_hx


# Train the model for (maximum) 50 epochs

history = train_model(lsun_model, train_generator, valid_generator, epochs=50)

# #### Plot the learning curves

# Run this cell to plot accuracy vs epoch and loss vs epoch

plt.figure(figsize=(15,5))
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


# You may notice overfitting in the above plots, through a growing discrepancy between the training and validation
# loss and accuracy. We will aim to mitigate this using data augmentation. Given our limited dataset, we may be
# able to improve the performance by applying random modifications to the images in the training data, effectively
# increasing the size of the dataset.

# #### Create a new data generator with data augmentation
# 
# Now write a function to create a new `ImageDataGenerator` object, which performs the following
# data preprocessing and augmentation:
# 
# * Scales the image pixel values by a factor of 1/255.
# * Randomly rotates images by up to 30 degrees
# * Randomly alters the brightness (picks a brightness shift value) from the range (0.5, 1.5)
# * Randomly flips images horizontally
# 
# Hint: you may need to refer to the [documentation](https://keras.io/preprocessing/image/#imagedatagenerator-class)
# for the `ImageDataGenerator`.

# Complete the following function. 

def get_ImageDataGenerator_augmented():
    return tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=30,
    brightness_range=(.5,1.5),
    horizontal_flip=True,
    rescale=(1/255)
    )

# Call the function to get an ImageDataGenerator as specified

image_gen_aug = get_ImageDataGenerator_augmented()


# Run this cell to define training and validation generators 

valid_generator_aug = get_generator(image_gen_aug, valid_dir)
train_generator_aug = get_generator(image_gen_aug, train_dir, seed=10)

# Reset the original train_generator with the same random seed

train_generator = get_generator(image_gen, train_dir, seed=10)


# #### Display sample augmented images and labels from the training set
# 
# The following depends on the function `get_generator` to be implemented correctly.
# If it raises an error, go back and check the function specifications carefully.
# 
# The cell will display augmented and non-augmented images (and labels) from the
# training dataset, using the `train_generator_aug` and `train_generator` objects defined
# above (if the images do not correspond to each other, check you have implemented the `seed` argument correctly).

# Display a few images and labels from the non-augmented and augmented generators

batch = next(train_generator)
batch_images = np.array(batch[0])
batch_labels = np.array(batch[1])

aug_batch = next(train_generator_aug)
aug_batch_images = np.array(aug_batch[0])
aug_batch_labels = np.array(aug_batch[1])

plt.figure(figsize=(16,5))
plt.suptitle("Unaugmented images", fontsize=16)
for n, i in enumerate(np.arange(10)):
    ax = plt.subplot(2, 5, n+1)
    plt.imshow(batch_images[i])
    plt.title(lsun_classes[np.where(batch_labels[i] == 1.)[0][0]])
    plt.axis('off')
plt.figure(figsize=(16,5))
plt.suptitle("Augmented images", fontsize=16)
for n, i in enumerate(np.arange(10)):
    ax = plt.subplot(2, 5, n+1)
    plt.imshow(aug_batch_images[i])
    plt.title(lsun_classes[np.where(aug_batch_labels[i] == 1.)[0][0]])
    plt.axis('off')


# Reset the augmented data generator

train_generator_aug = get_generator(image_gen_aug, train_dir)


# #### Train a new model on the augmented dataset

# Build and compile a new model

lsun_new_model = get_model((64, 64, 3))


# Train the model

history_augmented = train_model(lsun_new_model, train_generator_aug, valid_generator_aug, epochs=50)


# #### Plot the learning curves

# Run this cell to plot accuracy vs epoch and loss vs epoch

plt.figure(figsize=(15,5))
plt.subplot(121)
try:
    plt.plot(history_augmented.history['accuracy'])
    plt.plot(history_augmented.history['val_accuracy'])
except KeyError:
    try:
        plt.plot(history_augmented.history['acc'])
        plt.plot(history_augmented.history['val_acc'])
    except KeyError:
        plt.plot(history_augmented.history['categorical_accuracy'])
        plt.plot(history_augmented.history['val_categorical_accuracy'])
plt.title('Accuracy vs. epochs')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='lower right')

plt.subplot(122)
plt.plot(history_augmented.history['loss'])
plt.plot(history_augmented.history['val_loss'])
plt.title('Loss vs. epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper right')
plt.show() 


# Do you see an improvement in the overfitting?
# This will of course vary based on your particular run and whether you have altered
# the hyperparameters.

# #### Get predictions using the trained model

# Get model predictions for the first 3 batches of test data

num_batches = 3
seed = 25
test_generator = get_generator(image_gen_aug, test_dir, seed=seed)
predictions = lsun_new_model.predict(test_generator, steps=num_batches)


# Run this cell to view randomly selected images and model predictions

# Get images and ground truth labels
test_generator = get_generator(image_gen_aug, test_dir, seed=seed)
batches = []
for i in range(num_batches):
    batches.append(next(test_generator))
    
batch_images = np.vstack([b[0] for b in batches])
batch_labels = np.concatenate([b[1].astype(np.int32) for b in batches])

# Randomly select images from the batch
inx = np.random.choice(predictions.shape[0], 4, replace=False)
print(inx)

fig, axes = plt.subplots(4, 2, figsize=(16, 12))
fig.subplots_adjust(hspace=0.4, wspace=-0.2)

for n, i in enumerate(inx):
    axes[n, 0].imshow(batch_images[i])
    axes[n, 0].get_xaxis().set_visible(False)
    axes[n, 0].get_yaxis().set_visible(False)
    axes[n, 0].text(30., -3.5, lsun_classes[np.where(batch_labels[i] == 1.)[0][0]], 
                    horizontalalignment='center')
    axes[n, 1].bar(np.arange(len(predictions[i])), predictions[i])
    axes[n, 1].set_xticks(np.arange(len(predictions[i])))
    axes[n, 1].set_xticklabels(lsun_classes)
    axes[n, 1].set_title(f"Categorical distribution. Model prediction: {lsun_classes[np.argmax(predictions[i])]}")
    
plt.show()




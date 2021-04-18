# Import necessary libraries. Ensure that they are installed into your environment.

import tensorflow as tf
import pandas as pd
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from skimage.io import imread, imshow, imsave

# Set initial constants (e.g. image input size, number of object classes, etc.) #

h = 512
w = 512
dim = 3
num_classes = 22

EPOCHS = 1000
BATCH_SIZE = 5

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Set up object classes with their corresponding colors. The indices of each object class in the PASCAL image dataset match with those of their corresponding colors. #

color_list = [[0, 0, 0], [224, 224, 192], [128, 0, 0], [128, 0, 128], [192, 128, 128], [128, 192, 0], [0, 128, 0],
              [192, 0, 128], [192, 0, 0], [64, 0, 128], [128, 128, 0], [128, 64, 0], [64, 0, 0], [64, 128, 0],
              [0, 64, 128], [64, 128, 128], [0, 128, 128], [0, 0, 128], [128, 128, 128], [192, 128, 0], [0, 64, 0],
              [0, 192, 0]]
class_list = ['other', 'outline', 'aeroplane', 'bottle', 'person', 'train', 'bicycle', 'horse', 'chair', 'dog', 'bird',
              'sheep', 'cat', 'cow', 'TV/monitor', 'motorbike', 'bus', 'boat', 'car', 'dining table', 'potted plant',
              'sofa']

# Set paths to frames and mask directories #

train_frame_path = 'PATH\TO\TRAIN_FRAMES'
train_mask_path = 'PATH\TO\TRAIN_MASKS'
val_frame_path = 'PATH\TO\VAL_FRAMES'
val_mask_path = 'PATH\TO\VAL_MASKS'

# ============================================================================ Helper functions ============================================================================ #

def semantic_annotate(img):  # Annotate img pixelwise (assign integers to pixels)
    '''
    performs pixel-wise annotation of ground truth img
    
    semantic_annotate(img) takes in a ground truth segmentation map, img, in the form of a numpy array.
    One-hot encodes img (each pixel is turned into a numpy array, in which the bit at index i is a 1 if the pixel belongs to object class i, otherwise 0)
    Returns the one-hot encoding of img
    
    Parameters:
      img: numpy array (two dimensions for image size, third dimension for channels)
    
    Returns:
    numpy array
      One-hot encoding of img
    
    '''

    temp_img = img[:, :, :-1]
    new_img = np.zeros((temp_img.shape[0], temp_img.shape[1], num_classes))  # Create "empty" matrix

    for i in range(0, temp_img.shape[0]):  # Iterate across each column, for each row
        for j in range(0, temp_img.shape[1]):
            current_color = temp_img[i, j].tolist()  # Express 3-array at indices i, j as a 3-list
            current_color_index = color_list.index(current_color)
            new_img[i, j, current_color_index] = 1

    return new_img  # returns a (h, w, num_classes) annotation array


def img_from_annotate(img):  # Convert annotations back into images
    '''
    obtains a segmentation map from pixel-wise annotation (one-hot encoding) img
    
    img_from_annotate(img) takes in a one-hot encoding, img, in the form of a numpy array.
    Matches the one-hot encoding of each pixel in img to its corresponding color, based on object class.
    Returns the segmentation map that corresponds to the one-hot encoding, img
    
    Parameters:
      img: numpy array (two dimensions for image size, third dimension for one-hot encoded pixels)
    
    Returns:
    numpy array
      Segmentation map based on img
    
    '''
    temp_img = img[0]
    new_img = np.zeros((temp_img.shape[0], temp_img.shape[1], dim))
    for i in range(0, temp_img.shape[0]):
        for j in range(0, temp_img.shape[1]):
            z = temp_img[i, j].tolist()
            index = z.index(max(z))
            new_img[i, j] = color_list[index]

    return new_img  # returns a (h, w, dim) de-annotated array (actual RGB image)


def fill_class(pad):
    '''
    Sets the first element of each array along third dimensions of pad equal to 1
    
    fill_class(pad) takes in a three-dimensional array, pad, and changes the first element of each array along the third dimension of pad to a 1
    Returns modified pad
    Note: pad represents the necessary zero-padding for an image in order to resize it for the required model input size
    
    Parameters:
      pad: numpy array (two dimensions for image size, third dimension for one-hot encoded pixels)
    
    Returns:
    numpy array
      pad, modified to have the first element of each array along its third dimension set equal to 1
    
    '''
    
    for r in pad:
        for c in r:
            c[0] = 1

    return pad


def zero_pad(img):  # Zero-pad so that img dimensions fit (h, w, img.shape[2])
    '''
    Sets the first element of each array along third dimensions of pad equal to 1
    
    fill_class(pad) takes in a three-dimensional array, pad, and changes the first element of each array along the third dimension of pad to a 1
    Returns modified pad
    Note: pad represents the necessary zero-padding for an image in order to resize it for the required model input size
    
    Parameters:
      pad: numpy array (two dimensions for image size, third dimension for one-hot encoded pixels)
    
    Returns:
    numpy array
      pad, modified to have the first element of each array along its third dimension set equal to 1
    
    '''
    
    pad1 = np.zeros((img.shape[0], w - img.shape[1], img.shape[2]))
    pad2 = np.zeros((h - img.shape[0], w, img.shape[2]))
    if img.shape[2] == num_classes:
        pad1 = fill_class(pad1)
        pad2 = fill_class(pad2)
    img = np.concatenate([img, pad1], axis=1)
    img = np.concatenate([img, pad2], axis=0)

    return img  # returns zero-padded img


def truncate_list(l, n):
    '''
    Truncates a list, l, so that len(l) is divisible by n.
    
    truncate_list(l, n) takes in a list, l, and a positive integer, n, and truncates l so that the number of elements in l is divisible by n
    Returns the resized list, l
    
    Parameters:
      l: list (in this case, of strings that are filenames for images)
      n: int
    
    Returns:
    list (of str)
    
    '''
    
    new_size = (len(l) // n) * len(l)
    l = l[0: new_size]

    return l

# ========================================================================================================================================================================== #

# Obtain list of names in each frame or mask directory #

train_frame_list = os.listdir(train_frame_path)
train_mask_list = os.listdir(train_mask_path)
val_frame_list = os.listdir(val_frame_path)
val_mask_list = os.listdir(val_mask_path)

# The value of the 'n' parameter for truncate_list(), 500, is based on preference (can be changed).

train_frame_list = truncate_list(train_frame_list, 500)
train_mask_list = truncate_list(train_mask_list, 500)
val_frame_list = truncate_list(val_frame_list, 500)
val_mask_list = truncate_list(val_mask_list, 500)

activation = 'relu' # All convolutional layers in the U-Net model will have the specified activation function. Can be changed.
initializer = tf.keras.initializers.HeNormal() # Weights and biases initialized from He-Normal distribution. Can be changed.

# Build the U-Net Model #

inputs = tf.keras.layers.Input((h, w, dim)) # Input layer (model accepts inputs of dimensions h, w, dim)
s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs) # Normalize values in the input

# The model is comprised on several layer "blocks". Note the convolutional (Conv2D), max-pooling (MaxPooling2D), and dropout layers.
# Note: the purpose of Dropout is to prevent overfitting. This layer randomly removes a specified portion of the output from the previous convolutional layer.

C1 = tf.keras.layers.Conv2D(16, (3, 3), activation=activation, kernel_initializer=initializer, padding='same')(s)
C1 = tf.keras.layers.Dropout(0.1)(C1)
C1 = tf.keras.layers.Conv2D(16, (3, 3), activation=activation, kernel_initializer=initializer, padding='same')(C1)
P1 = tf.keras.layers.MaxPooling2D((2, 2))(C1)

C2 = tf.keras.layers.Conv2D(32, (3, 3), activation=activation, kernel_initializer=initializer, padding='same')(P1)
C2 = tf.keras.layers.Dropout(0.1)(C2)
C2 = tf.keras.layers.Conv2D(32, (3, 3), activation=activation, kernel_initializer=initializer, padding='same')(C2)
P2 = tf.keras.layers.MaxPooling2D((2, 2))(C2)

C3 = tf.keras.layers.Conv2D(64, (3, 3), activation=activation, kernel_initializer=initializer, padding='same')(P2)
C3 = tf.keras.layers.Dropout(0.2)(C3)
C3 = tf.keras.layers.Conv2D(64, (3, 3), activation=activation, kernel_initializer=initializer, padding='same')(C3)
P3 = tf.keras.layers.MaxPooling2D((2, 2))(C3)

C4 = tf.keras.layers.Conv2D(128, (3, 3), activation=activation, kernel_initializer=initializer, padding='same')(P3)
C4 = tf.keras.layers.Dropout(0.2)(C4)
C4 = tf.keras.layers.Conv2D(128, (3, 3), activation=activation, kernel_initializer=initializer, padding='same')(C4)
P4 = tf.keras.layers.MaxPooling2D((2, 2))(C4)

C5 = tf.keras.layers.Conv2D(256, (3, 3), activation=activation, kernel_initializer=initializer, padding='same')(P4)
C5 = tf.keras.layers.Dropout(0.3)(C5)
C5 = tf.keras.layers.Conv2D(256, (3, 3), activation=activation, kernel_initializer=initializer, padding='same')(C5)
P5 = tf.keras.layers.MaxPooling2D((2, 2))(C5)

# Upscaling #

# Recall information that was lost during the max-pooling phases. However, we are still randomly omitting certain data to prevent overfitting.

U6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(C5)
U6 = tf.keras.layers.concatenate([U6, C4])
C6 = tf.keras.layers.Conv2D(128, (3, 3), activation=activation, kernel_initializer=initializer, padding='same')(U6)
C6 = tf.keras.layers.Dropout(0.3)(C6)
C6 = tf.keras.layers.Conv2D(128, (3, 3), activation=activation, kernel_initializer=initializer, padding='same')(C6)

U7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(C6)
U7 = tf.keras.layers.concatenate([U7, C3])
C7 = tf.keras.layers.Conv2D(64, (3, 3), activation=activation, kernel_initializer=initializer, padding='same')(U7)
C7 = tf.keras.layers.Dropout(0.2)(C7)
C7 = tf.keras.layers.Conv2D(64, (3, 3), activation=activation, kernel_initializer=initializer, padding='same')(C7)

U8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(C7)
U8 = tf.keras.layers.concatenate([U8, C2])
C8 = tf.keras.layers.Conv2D(32, (3, 3), activation=activation, kernel_initializer=initializer, padding='same')(U8)
C8 = tf.keras.layers.Dropout(0.2)(C8)
C8 = tf.keras.layers.Conv2D(32, (3, 3), activation=activation, kernel_initializer=initializer, padding='same')(C8)

U9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(C8)
U9 = tf.keras.layers.concatenate([U9, C1])
C9 = tf.keras.layers.Conv2D(16, (3, 3), activation=activation, kernel_initializer=initializer, padding='same')(U9)
C9 = tf.keras.layers.Dropout(0.1)(C9)
C9 = tf.keras.layers.Conv2D(16, (3, 3), activation=activation, kernel_initializer=initializer, padding='same')(C9)

outputs = tf.keras.layers.Conv2D(num_classes, (1, 1), activation='sigmoid')(C9) # Output (final) layer is a convolutional layer.

# Compile the model with optimizer and loss function #

# Note: the optimizer is responsible for adjusting weights/biases to obtain best possible results.
# We use the binary cross-entropy function for computing loss, as the model's output is a prediction for a one-hot encoding of an image.

optimizer = tf.keras.optimizers.Adam(lr=0.001)
loss = tf.keras.losses.BinaryCrossentropy()

# Create an instance of the U-Net model #

model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Allocate numpy arrays for storing batch data #

train_frame_array = np.zeros((BATCH_SIZE, h, w, dim))  # Zero-padded JPEG images
train_mask_array = np.zeros((BATCH_SIZE, h, w, num_classes))  # Zero-padded, annotated segmentations
val_frame_array = np.zeros((BATCH_SIZE, h, w, dim))  # Zero-padded JPEG images
val_mask_array = np.zeros((BATCH_SIZE, h, w, num_classes))  # Zero-padded, annotated segmentations

# Initialize variables for early-stopping #

current_loss_minimum = 1
loss_convergence_streak = 0
patience = 25
losses = []
max_epoch = 0

# Training loop #

for epoch in range(EPOCHS):  # Iterate over total number of epochs
    print('\nEpoch {}:'.format(epoch + 1))
    # start_time = time()
    trained_datapoint_num = 0  # Keep track of number of images on which model has trained

    batch_losses = []
    for batch in range(len(train_frame_list) // BATCH_SIZE):  # Iterate over each batch
        original_sizes = []  # Keep track of original image sizes to calculate true loss value

        for i in range(BATCH_SIZE):  # Fill training arrays with data for each image in batch
            x = imread(train_frame_path + train_frame_list[i + trained_datapoint_num])
            y = imread(train_mask_path + train_mask_list[i + trained_datapoint_num])
            original_sizes.append(x.shape)
            y = semantic_annotate(y)  # y-dim: (y.shape[0], y.shape[1], num_classes)
            x = zero_pad(x)  # x-dim: (h, w, dim)
            y = zero_pad(y)  # y-dim: (h, w, num_classes)
            train_frame_array[i] = x
            train_mask_array[i] = y
        trained_datapoint_num += BATCH_SIZE  # Update number of images on which model has trained

        with tf.GradientTape() as tape:  # Open up the gradient tape scope
            logits = model(train_frame_array, training=True)  # Call model on batch training frames

            for p in range(BATCH_SIZE):
                temp_frame = logits[p][0:original_sizes[p][0], 0:original_sizes[p][1]]
                temp_mask = train_mask_array[p][0:original_sizes[p][0], 0:original_sizes[p][1]]
                loss_value = loss(temp_mask, temp_frame)  # Compute loss value for batch training frames and masks

        batch_losses.append(loss_value)
        grads = tape.gradient(loss_value, model.trainable_weights) # Compute gradient vector of the loss function (with respect to weights/biases)
        optimizer.apply_gradients(zip(grads, model.trainable_weights)) # Adjust weights and biases of network
        print('Training loss for Batch {}: {}'.format(batch + 1, float(loss_value)))
        max_epoch += 1
        if epoch % 25 == 0:
            model.save('segment_model') # Save the model (weights and biases)

    average_batch_loss = sum(batch_losses) / (len(train_frame_list) // BATCH_SIZE)
    losses.append(average_batch_loss)
    print('Average batch training loss for Epoch {}: {}'.format(epoch + 1, average_batch_loss))
    if average_batch_loss >= current_loss_minimum * 0.9999: # Early stopping: if model makes little to no progress after some time, terminate training early.
        loss_convergence_streak += 1
        if loss_convergence_streak == patience:
            break
    current_loss_minimum = average_batch_loss

# The next time you want to resume training, use this line of code instead of creating a new instance of the model. #

model = tf.keras.models.load_model('seg_car')

# The following code is for the purpose of seeing the model's prediction for segmenting a single image. #

# img = imread(train_frame_path + '2008_000074.png')
# mask = imread(train_mask_path + '2008_000074.png')
# x = img.shape[0]
# y = img.shape[1]
# mask = semantic_annotate(mask)
# img = zero_pad(img)
# mask = zero_pad(mask)
# img = img[None, :, :, :]
# predict = model.predict(img)
# p = predict[0]
# predict = img_from_annotate(predict)
# new_predict = predict[0:x, 0:y]
# print(new_predict.shape)
# plt.figure()
# plt.imshow(new_predict.astype(np.uint8))
# plt.show()






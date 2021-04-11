# -*- coding: utf-8 -*-
"""Facial Detection CNN

# **1. Importing Necessary Files and Libraries**
"""

# importing python file containing CNN architecture
from smallervggnet import SmallerVGGNet
from cnn_first import Simple_CNN

# importing the necessary packages
from sklearn.preprocessing import LabelBinarizer
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.models import load_model
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
import cv2
import os

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use('Agg')

# defining paths for saving results and image sizes
SAVE_PATH_VGG = 'drive/My Drive/Facial Classification - Gender/VGG_results'
SAVE_PATH_CNN = 'drive/My Drive/Facial Classification - Gender/SimpleCNN_results'
IMG_WIDTH = 64
IMG_HEIGHT = 64

"""# **2. Loading Images**"""

# initialize the data and labels
print('[INFORMATION] Loading images...')
data = []
labels = []

# grab the images and randomly shuffle them
DATASET_PATH = 'drive/My Drive/Facial Classification - Gender/dataset_genre'
imagePaths = sorted(list(paths.list_images(DATASET_PATH)))
random.seed(42)
random.shuffle(imagePaths)

# loop over the input images
for imagePath in imagePaths:
  # load the image, resize it to 64 x 64 pixels (the required input spatial dimensions of SmallerVGGNet),
  # and store it in the data list
  image = cv2.imread(imagePath)
  image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
  data.append(image)

  # extract the class label from the image path and update the labels list
  label = imagePath.split(os.path.sep)[-2]
  labels.append(label)

print('[INFORMATION] Images Loaded and Labels Extracted sucessfully!')

print('Dataset size: {} images of 2 labels'.format(len(data)))

"""# **3. Data Preprocessing and Data Augmentation**"""

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype = 'float')/255.0
labels = np.array(labels)

# partition the data into training and testing splits using 75 % of the data for training and the remaining 25 % for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size = 0.25, random_state=42)

# convert the labels from strings to integers
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

# now, convert the labels from integers to vectors
trainY = to_categorical(trainY, num_classes=2, dtype='float32')
testY = to_categorical(testY, num_classes=2, dtype='float32')

# construct image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                         height_shift_range = 0.1, shear_range = 0.2,
                         zoom_range = 0.2, horizontal_flip = True,
                         fill_mode = 'nearest')

"""# **4. First Model Initializing - VGG Architecture**"""

# initialize our VGG-lie Convolutional Neural Network
model = SmallerVGGNet.build(width=IMG_WIDTH, height=IMG_HEIGHT, depth=3, classes=len(lb.classes_))

# initialize our initial learning rage, # of epochs to train for and batch size
INIT_LR = 0.01
EPOCHS = 75
BATCH_SIZE = 32

# initialize the model and optimizer
print('[INFORMATION] Loading Neural Network Model...')
opt = SGD(lr = INIT_LR, decay = INIT_LR/EPOCHS)
model.compile(loss='categorical_crossentropy', optimizer = opt, metrics=['accuracy'])
print('[INFORMATION] Neural Network Model successfully loaded!\n')
model.summary()

"""# **5. First Model Training**"""

# train the network
print('[INFORMATION] Starting First Model Training...')
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BATCH_SIZE),
                        validation_data = (testX, testY),
                        steps_per_epoch = len(trainX)//BATCH_SIZE,
                        epochs = EPOCHS)
print('[INFORMATION] First CNN Model sucessfully trained')

"""# **6. First Model Evaluation**"""

# evaluate the network
print('[INFORMATION] Starting Neural Network Evaluation...')
predictions = model.predict(testX, batch_size = 32)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis = 1),
                            target_names = lb.classes_))

# plot the training loss and accuracy curves
N = np.arange(0, EPOCHS)
plt.style.use('ggplot')
plt.figure(figsize=(20,20))
plt.plot(N, H.history['loss'], label = 'train_loss')
plt.plot(N, H.history['val_loss'], label = 'val_loss')
plt.plot(N, H.history['accuracy'], label = 'train_acc')
plt.plot(N, H.history['val_accuracy'], label = 'val_acc')
plt.title('Training Loss and Accuracy (SmallerVGGNet)')
plt.xlabel('Epoch #')
plt.ylabel('Loss / Accuracy')
plt.legend()
plt.show()
plt.savefig(SAVE_PATH_VGG + '/learning_curves.png')

# save the model and label binarizer to disk
print('[INFORMATION] Serializing Network and Label Binarizer...')
model.save(SAVE_PATH_VGG)

"""# **7. Second Model Initializing - Double Convolution Architecture**"""

# initialize our Simple Convolutional Neural Network
model = Simple_CNN.build(width=IMG_WIDTH, height=IMG_HEIGHT, depth=3, classes=len(lb.classes_))

# initialize our initial learning rage, # of epochs to train for and batch size
EPOCHS = 150
BATCH_SIZE = 32

# initialize the model and optimizer
print('[INFORMATION] Loading Neural Network Model...')
model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])
print('[INFORMATION] Neural Network Model successfully loaded!\n')
model.summary()

"""# **8. Second Model Training**"""

# train the network
print('[INFORMATION] Starting Second Model Training...')
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BATCH_SIZE),
                        validation_data = (testX, testY),
                        steps_per_epoch = len(trainX)//BATCH_SIZE,
                        epochs = EPOCHS)
print('[INFORMATION] Second CNN Model sucessfully trained')

"""# **9. Second Model Evaluating**"""

# evaluate the network
print('[INFORMATION] Starting Neural Network Evaluation...')
predictions = model.predict(testX, batch_size = 32)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis = 1),
                            target_names = lb.classes_))

# plot the training loss and accuracy curves
N = np.arange(0, EPOCHS)
plt.style.use('ggplot')
plt.figure(figsize=(10,10))
plt.plot(N, H.history['loss'], label = 'train_loss')
plt.plot(N, H.history['val_loss'], label = 'val_loss')
plt.plot(N, H.history['accuracy'], label = 'train_acc')
plt.plot(N, H.history['val_accuracy'], label = 'val_acc')
plt.title('Training Loss and Accuracy (Simple CNN)')
plt.xlabel('Epoch #')
plt.ylabel('Loss / Accuracy')
plt.legend()
plt.show()
plt.savefig(SAVE_PATH_CNN + '/learning_curve.png')

# save the model and label binarizer to disk
print('[INFORMATION] Serializing Network and Label Binarizer...')
LABEL_BIN = SAVE_PATH_CNN + '/binarizer.txt'
model.save(SAVE_PATH_CNN)

"""# **10. First Model Testing**"""

# load the desired model and label binarizer
print('[INFORMATION] Loading Neural Network Model and Label Binarizer')
model = load_model(SAVE_PATH_VGG)

# load images to predict
PREDICT_PATH = 'drive/My Drive/Facial Classification - Gender/dataset_test'
imagePaths = sorted(list(paths.list_images(PREDICT_PATH)))

plt.figure(figsize=(10,10))
plt.subplots_adjust(wspace = 0.2, hspace = 0.01)
k = 1
for imagePath in imagePaths:
  # load the image, resize it to 64 x 64 pixels, and store it in the data list
  image = cv2.imread(imagePath)
  output = image.copy()
  image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))

  # add the batch dimension
  image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

  # make a prediction on the image
  preds = model.predict(image)

  # find the class label index with the largest corresponding probability
  i = preds.argmax(axis = 1)[0]
  if i == 0:
    label = 'Man'
  else:
    label = 'Woman'

  # draw the class label + probability on the output image
  text = '{}: {:.2f} %'.format(label, preds[0][i]*100)

  # show the output imges
  plt.subplot(2, 5, k)
  plt.imshow(output)
  plt.title(text, fontsize = 10)
  plt.grid(b=None)
  plt.axis('off')

  k += 1

plt.savefig(SAVE_PATH_VGG + '/predictions.png')

"""# **11. Second Model Testing**"""

# load the desired model and label binarizer
print('[INFORMATION] Loading Neural Network Model and Label Binarizer')
model = load_model(SAVE_PATH_CNN)

# load images to predict
PREDICT_PATH = 'drive/My Drive/Facial Classification - Gender/dataset_test'
imagePaths = sorted(list(paths.list_images(PREDICT_PATH)))

plt.figure(figsize=(10,10))
plt.subplots_adjust(wspace = 0.2, hspace = 0.01)
k = 1
for imagePath in imagePaths:
  # load the image, resize it to 64 x 64 pixels, and store it in the data list
  image = cv2.imread(imagePath)
  output = image.copy()
  image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))

  # add the batch dimension
  image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

  # make a prediction on the image
  preds = model.predict(image)

  # find the class label index with the largest corresponding probability
  i = preds.argmax(axis = 1)[0]
  if i == 0:
    label = 'Man'
  else:
    label = 'Woman'

  # draw the class label + probability on the output image
  text = '{}: {:.2f} %'.format(label, preds[0][i]*100)

  # show the output imges
  plt.subplot(2, 5, k)
  plt.imshow(output)
  plt.title(text, fontsize = 10)
  plt.grid(b=None)
  plt.axis('off')

  k += 1

plt.savefig(SAVE_PATH_CNN + '/predictions.png')
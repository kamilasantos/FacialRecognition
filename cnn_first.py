# import the necessary packages
import numpy as np
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Activation
from keras import backend as K

class Simple_CNN:
    @staticmethod
    def build(width, height, depth, classes, finalAct = "softmax"):

        InputShape = (height, width, depth)
        chanDim = -1

        # if we are using "channels first", update the input shape
        # and channels dimension
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        model = Sequential()

        # step 1 - adding the convolutional layer
        model.add(Convolution2D(32, (3,3), input_shape = InputShape, activation = 'relu'))

        # step 2 - adding the pooling layer
        model.add(MaxPooling2D(pool_size=(2,2)))

        # step 2.1 - adding a second convolutional layer and a second pooling layer
        model.add(Convolution2D(32, (3,3), activation = 'relu'))
        model.add(MaxPooling2D(pool_size = (2,2)))

        # step 3 - adding the flattening layer
        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation("relu"))
        
        # step 4 - adding the fully connected layer
        model.add(Dense(classes))
        model.add(Activation(finalAct))

        return model

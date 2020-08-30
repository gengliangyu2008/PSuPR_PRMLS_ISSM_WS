from tensorflow.keras.utils import to_categorical
import numpy as np

# Load the data
trDat = np.load('./../kmnist-train-imgs.npz')['arr_0']
trLbl = np.load('./../kmnist-train-labels.npz')['arr_0']
tsDat = np.load('./../kmnist-test-imgs.npz')['arr_0']
tsLbl = np.load('./../kmnist-test-labels.npz')['arr_0']

# Convert the data into 'float32'
# Rescale the values from 0~255 to 0~1
trDat = trDat.astype('float32') / 255
tsDat = tsDat.astype('float32') / 255

# Retrieve the row size of each image
# Retrieve the column size of each image
imgrows = trDat.shape[1]
imgclms = trDat.shape[2]

# reshape the data to be [samples][width][height][channel]
# This is required by Keras framework
trDat = trDat.reshape(trDat.shape[0], imgrows, imgclms, 1)
tsDat = tsDat.reshape(tsDat.shape[0], imgrows, imgclms, 1)

# Perform one hot encoding on the labels
# Retrieve the number of classes in this problem
trLbl = to_categorical(trLbl)
tsLbl = to_categorical(tsLbl)
num_classes = tsLbl.shape[1]

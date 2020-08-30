# -*- coding: utf-8 -*-
# refactored from original wks5_5_start.py

import numpy as np
import sklearn.metrics as metrics

from keras.callbacks import ModelCheckpoint, CSVLogger

# fix random seed for reproducibility
from kmnist.data_reading import tsDat, trDat, trLbl, tsLbl, num_classes
from kmnist.plot_utils import plt_model_csv
from kmnist.seq_model import createModel

np.random.seed(29)

model_name = 'kmnist_sequential_model'

# Setup the models
model = createModel(num_classes)  # This is meant for training
modelGo = createModel(num_classes)  # This is used for final testing

model.summary()

# Create checkpoint for the training
# This checkpoint performs model saving when an epoch gives highest testing accuracy
filepath = model_name + ".hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')

# Log the epoch detail into csv
csv_logger = CSVLogger(model_name + '.csv')
callbacks_list = [checkpoint, csv_logger]

# Fit the model, this is where the training starts
model.fit(trDat, trLbl, validation_data=(tsDat, tsLbl), epochs=20, batch_size=128, callbacks=callbacks_list)

# Now the training is complete, we get another object to load the weights compile it,
# so that we can do final evaluation on it
modelGo.load_weights(filepath)
modelGo.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Make classification on the test dataset
predicts = modelGo.predict(tsDat)

# Prepare the classification output
# for the classification report
predout = np.argmax(predicts, axis=1)
testout = np.argmax(tsLbl, axis=1)

# the labels for the classification report
labelname = ['お O', 'き Ki', 'す Su', 'つ Tsu', 'な Na', 'は Ha', 'ま Ma', 'や Ya', 'れ Re', 'を Wo']

testScores = metrics.accuracy_score(testout, predout)
confusion = metrics.confusion_matrix(testout, predout)

print("Best accuracy (on testing dataset): %.2f%%" % (testScores * 100))
print(metrics.classification_report(testout, predout, target_names=labelname, digits=4))
print(confusion)

plt_model_csv(model_name)
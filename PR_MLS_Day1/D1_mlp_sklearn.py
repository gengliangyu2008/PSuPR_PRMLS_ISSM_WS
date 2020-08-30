#!/usr/bin/env python
# coding: utf-8

# ## MLP using Sklearn

from sklearn.preprocessing import StandardScaler
from sklearn import metrics 
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn.model_selection import train_test_split
import numpy


# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)


# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
y = dataset[:,8]


# train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/3,random_state=42, stratify=y)


# Normalization
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)  


from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(10,10), max_iter=1000, verbose=2)  
mlp.fit(X_train, y_train)  


predictions = mlp.predict(X_test)


print("Accuracy", metrics.accuracy_score(y_test, predictions))
print(confusion_matrix(y_test,predictions))  
print(classification_report(y_test,predictions))  


import matplotlib.pyplot as plt
# only available in jupyter, not in py codes
# get_ipython().run_line_magic('matplotlib', 'inline')

plt.plot(mlp.loss_curve_)
plt.title("NN Loss Curve")
plt.xlabel("number of steps")
plt.ylabel("loss function")
plt.show()


mlp.intercepts_[0]


mlp.coefs_[0]
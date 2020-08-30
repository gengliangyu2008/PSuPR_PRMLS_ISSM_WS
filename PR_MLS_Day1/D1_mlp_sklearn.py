#!/usr/bin/env python
# coding: utf-8

# ## MLP using Sklearn

from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix 

from D1_data_load import X_train, y_train, X_test, y_test

from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(10,10), max_iter=800, verbose=2)
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


print("mlp.intercepts_[0]", mlp.intercepts_[0])

print("mlp.coefs_[0]", mlp.coefs_[0])

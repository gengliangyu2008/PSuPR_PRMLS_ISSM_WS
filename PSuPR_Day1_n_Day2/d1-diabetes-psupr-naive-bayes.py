from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from dataUtils import X_train, y_train, X_test, y_test

# Naive Bayes


# Initiating the Gaussian Classifier
mod = GaussianNB()

# Training your model
mod.fit(X_train, y_train)

# Predicting Outcome
predicted = mod.predict(X_test)

mod.score(X_test,y_test)

# Confusion Matrix
y_pred = mod.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# You can compare the performance of multiple models in one ROC chart.
# Wrtie your own codes in the cells below.
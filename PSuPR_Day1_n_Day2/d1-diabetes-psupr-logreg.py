import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from dataUtils import X_train, y_train, X_test, y_test

logreg = LogisticRegression(C=0.01).fit(X_train, y_train)
print("Training set score: {:.3f}".format(logreg.score(X_train, y_train)))
print("Test set score: {:.3f}".format(logreg.score(X_test, y_test)))
print(logreg.intercept_.T)
print(logreg.coef_.T)

# Confusion Matrix

y_pred = logreg.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ROC
from sklearn import metrics

print("Accuracy=", metrics.accuracy_score(y_test, y_pred))

y_pred_proba = logreg.predict_proba(X_test)[::, 1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)

plt.plot(fpr, tpr, label="logreg, auc=" + str(auc))
plt.plot([0, 1], [0, 1], 'k--')
plt.legend(loc=4)
plt.show()

# Gains / Lift Chart

# !conda install -c conda-forge scikit-plot
import matplotlib.pyplot as plt
import scikitplot as skplt
y_pred_probas = logreg.predict_proba(X_test)
skplt.metrics.plot_cumulative_gain(y_test, y_pred_probas)
plt.show()
skplt.metrics.plot_lift_curve(y_test, y_pred_probas)
plt.show()

import pandas as pd
import numpy as np
import pickle

import matplotlib.pyplot as plt
import seaborn as sns
from ggplot import *

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import neighbors
from sklearn.naive_bayes import GaussianNB

def log_reg():
	LogReg = LogisticRegression()
	LogReg.fit(X_train, y_train)
	## Logistic regression
	print("Logistic regression")

	LogReg = LogisticRegression()
	LogReg.fit(X_train, y_train)
	y_pred = LogReg.predict(X_test)

	confusion_matrix = confusion_matrix(y_test, y_pred)
	print(confusion_matrix)
	print(classification_report(y_test, y_pred))


#KNN

knn = neighbors.KNeighborsClassifier(n_neighbors = 1)
knn_model = knn.fit(X_train, y_train)
print('KNN accuracy for train set: %f' % knn_model.score(X_train, y_train))
print('KNN accuracy for test set: %f' % knn_model.score(X_test, y_test))

#Naive Bayes

nb = GaussianNB()
nb_model = nb.fit(X_train,y_train)
print('Naïve Bayes accuracy for train set: %f' % nb_model.score(X_train, y_train))
print('Naïve Bayes accuracy for test set: %f' % nb_model.score(X_test, y_test))

print("Logistic regression")

#plt.subplots(figsize=(10,10))
#ax = plt.axes()
#ax.set_title("Titanic Heatmap")
#corr = X_train.corr()
#sns.heatmap(corr,xticklabels=corr.columns.values,yticklabels=corr.columns.values,annot=True, cmap="coolwarm")
#plt.show()
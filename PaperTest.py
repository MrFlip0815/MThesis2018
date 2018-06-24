
# MLP testing


import sys
import optunity # svm tuning
import optunity.metrics
import os
#from sklearn.svm import SVC
from sklearn import preprocessing
#from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

file = os.path.join("data/","kwon_combo.csv")

import pandas as pd
if sys.platform == 'win32':
        wine = pd.read_csv(file,
                            	names = [
				"filename",
				"gender",
				"k1", 
				"k2",
				"k3", 
				"k4", 
				"k5"])

X = wine.drop('gender', axis=1)
X = X.drop('filename', axis=1)

y = wine['gender']


X_train, X_test, y_train, y_test = train_test_split(X,y)

scaler = preprocessing.StandardScaler()
scaler_robust = preprocessing.RobustScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)

mlp = MLPClassifier(hidden_layer_sizes=(5,12,12),max_iter=5000)


print(mlp.fit(X_train,y_train))


predictions = mlp.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix

print(confusion_matrix(y_test,predictions))

print(classification_report(y_test,predictions))
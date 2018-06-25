import os
import sys
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

import numpy

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit


#training_file = os.path.join("data/","kwon_combo.csv")

training_file = os.path.join("full/","KWON.LOBO.FULL.csv")

# CSV files for validation - all same content
validation_FG_file = os.path.join("data/","FGNET.KWON.combo.csv")
validation_CK_file = os.path.join("data/","CK.KWON.combo.csv")
validation_CalTech_file = os.path.join("data/","CalTech.KWON.combo.csv")
validation_FEI_file = os.path.join("data/","FEI.KWON.combo.csv")

validation_FEI_KWONLOBO = os.path.join("data/","FEI.KWON.LOBO.combo.csv")

# let's try FG first
validation_file = validation_CalTech_file

training_data = pd.read_csv(training_file, 
				names = [
				"filename",
				"gender",
				"k1", 
				"k2",
				"k3", 
				"k4", 
				"k5",
				"k6",
				])

validation_data = pd.read_csv(validation_FEI_KWONLOBO, 
				names = [
				"filename",
				"gender",
				"k1", 
				"k2",
				"k3", 
				"k4", 
				"k5",
				"k6",
				])

training_data['gender'] = training_data['gender'].map({'female': 1, 'male': 0})
validation_data['gender'] = validation_data['gender'].map({'female': 1, 'male': 0})


validation_data = validation_data.sample(frac=1).reset_index(drop=True)
#https://github.com/keras-team/keras/issues/1478
########### KERAS MODEL  
#					https://datascience.stackexchange.com/questions/12532/does-batch-size-in-keras-have-any-effects-in-results-quality
# loss und accuracy https://stackoverflow.com/questions/34518656/how-to-interpret-loss-and-accuracy-for-a-machine-learning-model
# AdaBoost http://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_twoclass.html
model = Sequential()
model.add(Dense(units=8, activation='relu', input_dim=6))
model.add(Dense(units=10, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units=5, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

###########

# drop unwanted columns
X_training = training_data.drop(['gender','filename'], axis=1)
# this is the class label 0 or 1
y_training = training_data['gender']


X_test = validation_data.drop(['gender','filename'], axis=1)
y_test = validation_data['gender']


# ADABOOST


#X_train_ada, X_test_ada, y_train_ada, y_test_ada = train_test_split(X_training, y_training, test_size=0.2, random_state=0)


#http://scikit-learn.org/stable/modules/cross_validation.html

clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),algorithm="SAMME.R", n_estimators=200 )
cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=0)
scores = cross_val_score(clf, X_training, y_training, cv=cv)

print(scores)

#for estimator in clf.estimators_:
#    print(estimator.predict(X_test))




##########




print("SYS EXIT")
sys.exit(0)


# train full set
history_train = model.fit(X_training, y_training, epochs=1500, batch_size=128)
#evaluate model 

score,acc = model.evaluate(X_test,y_test, batch_size=128)
print(score,acc)

#print(history.history.keys())

from keras.utils import plot_model
plot_model(model, to_file='model.png')

# Beispiel 
# https://www.codesofinterest.com/2017/03/graph-model-training-history-keras.html

import matplotlib.pyplot as plt  

plt.figure(1)  
   
 # summarize history for accuracy  
plt.subplot(211)  
plt.plot(history_train.history['loss'])  
plt.plot(history_train.history['acc'])  
plt.title('model accuracy')  
plt.ylabel('accuracy')  
plt.xlabel('epoch')  
plt.legend(['loss', 'acc'], loc='upper left')  

plt.show()
   
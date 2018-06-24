import os
import sys
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

import numpy

training_file = os.path.join("data/","kwon_combo.csv")
# CSV files for validation - all same content
validation_FG_file = os.path.join("data/","FGNET.KWON.combo.csv")
validation_CK_file = os.path.join("data/","CK.KWON.combo.csv")
validation_CalTech_file = os.path.join("data/","CalTech.KWON.combo.csv")
validation_FEI_file = os.path.join("data/","FEI.KWON.combo.csv")

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
				"k5"])

validation_data = pd.read_csv(validation_file, 
				names = [
				"filename",
				"gender",
				"k1", 
				"k2",
				"k3", 
				"k4", 
				"k5"])


validation_data['gender'] = validation_data['gender'].map({'female': 1, 'male': 0})

validation_data = validation_data.sample(frac=1).reset_index(drop=True)

print(validation_data)

########### KERAS MODEL  
# loss und accuracy https://stackoverflow.com/questions/34518656/how-to-interpret-loss-and-accuracy-for-a-machine-learning-model
model = Sequential()
model.add(Dense(units=6, activation='relu', input_dim=5))
model.add(Dense(units=12, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units=12, activation='relu'))
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


# train full set
history_train = model.fit(X_test, y_test, epochs=5000, batch_size=16)
#evaluate model 

score,acc = model.evaluate(X_test,y_test, batch_size=16)
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
   
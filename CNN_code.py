import numpy as np
import sys, os
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

# data preprocessing
print("[*] Start")

datasets = np.loadtxt('파일.csv', delimiter=',',skiprows=1)
xy_data = datasets   # exclude header
train_set, test_set = train_test_split(xy_data, test_size=0.3, random_state=123)   
print('Training Length : ', len(train_set), 'Test Length : ', len(test_set))

# 7 x 7
x_train_data = train_set[:,:피처 갯수].reshape(-1,7,7,1)
y_train_data = train_set[:,피처 갯수:]

# 7 x 7
x_test_data = test_set[:,:피처 갯수].reshape(-1,7,7,1)
y_test_data = test_set[:,피처 갯수:]

print("[x_train_data]",x_train_data.shape)
print("[y_train_data]", y_train_data.shape)
print("[x_test_data]", x_test_data.shape)
print("[y_test_data]", y_test_data.shape)

y_train_data = np_utils.to_categorical(y_train_data, 2)
y_test_data = np_utils.to_categorical(y_test_data, 2)

#======== model start ===========
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from keras.utils import np_utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

import matplotlib.pyplot as plt
import os
import numpy

# 7 x 7
X_train = x_train_data
Y_train = y_train_data

# 7 x 7
X_validation = x_test_data
Y_validation = y_test_data 

model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3), input_shape=(7,7,1), activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.25))
model.add(Flatten())

model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))

#model.add(Dense(2, activation='softmax'))
model.add(Dense(2, activation='sigmoid'))

model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])

history = model.fit(X_train, Y_train,validation_data=(X_validation, Y_validation),epochs=25, batch_size=200)
print('train_Accuracy : {:.4f}'.format(model.evaluate(X_train, Y_train)[1]))
print('test_Accuracy: {:.4f}'.format(model.evaluate(X_validation, Y_validation)[1]))

# 그래프 코드
y_vloss = history.history['val_loss']
y_loss = history.history['loss']

x_len = numpy.arange(len(y_loss))
plt.plot(x_len, y_loss, marker='.', c='blue', label="Train-set Loss")
plt.plot(x_len, y_vloss, marker='.', c='red', label="Validation-set Loss")

plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
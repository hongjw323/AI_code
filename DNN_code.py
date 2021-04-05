from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import metrics
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import datasets              
from keras.utils import np_utils
from tensorflow.keras.layers import Dropout
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

print("[*] Start")

datasets = np.loadtxt('파일.csv', delimiter=',',skiprows=1)
xy_data = datasets
train_set, test_set = train_test_split(xy_data, test_size=0.3, random_state=123)
print('Training Length : ', len(train_set), 'Test Length : ', len(test_set))

x_train_data = train_set.T[:피처 갯수]
y_train_data = train_set.T[피처 갯수]

x_test_data = test_set.T[:피처 갯수]
y_test_data = test_set.T[피처 갯수]

print("[x_train_data]",x_train_data.shape)
print("[y_train_data]", y_train_data.shape)
print("[x_test_data]", x_test_data.shape)
print("[y_test_data]", y_test_data.shape)

#modeling
model = models.Sequential()
#input layer
model.add(layers.Dense(피처 갯수, activation='relu', input_shape=(피처 갯수,)))

# hidden layer
model.add(layers.Dense(15, activation='relu'))
model.add(layers.Dense(7, activation='relu'))
model.add(Dropout(0.2))


#out layer
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])

hist = model.fit(x_train_data.T,y_train_data.T, epochs=25,batch_size=200,
                    validation_data = (x_test_data.T, y_test_data.T))

#  ####################
performance_train = model.evaluate(x_train_data.T, y_train_data.T)
print('train_Accuracy : ',format(performance_train[1]))
performance_test = model.evaluate(x_test_data.T, y_test_data.T)
print('test_Accuracy : ',format(performance_test[1]))
#print('Test Loss and Accuracy ->', performace_test)

# 그래프 코드
fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
loss_ax.legend(loc='upper right')

acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')
acc_ax.plot(hist.history['val_accuracy'], 'g', label='val acc')
acc_ax.set_ylabel('accuracy')
acc_ax.legend(loc='upper left')

plt.show()
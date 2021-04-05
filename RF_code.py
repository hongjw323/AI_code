import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from keras import models


# data preprocessing
print("[*] Start : RandomForest Classification")

datasets = np.loadtxt('파일.csv', delimiter=',',skiprows=1)
xy_data = datasets   # exclude header
train_set, test_set = train_test_split(xy_data, test_size=0.3, random_state=123)   
print('Training Length : ', len(train_set), 'Test Length : ', len(test_set))

x_train_data = train_set[:,1:]
y_train_data = train_set[:,:1].reshape(-1,)

x_test_data = test_set[:,1:]
y_test_data = test_set[:,:1].reshape(-1,)

print("[x_train_data]",x_train_data.shape)
print("[y_train_data]", y_train_data.shape)
print("[x_test_data]", x_test_data.shape)
print("[y_test_data]", y_test_data.shape)

rf = RandomForestClassifier(n_estimators=60, random_state=123)
rf.fit(x_train_data, y_train_data)

print("train_accuracy :",rf.score(x_train_data, y_train_data))
print("\n")

print("test_accuracy :",rf.score(x_test_data, y_test_data))
print("\n")
y_pred = rf.predict(x_test_data)
print(y_pred)

print(confusion_matrix(y_test_data, y_pred))

#confusion matrix (1:악성, 0:정상)
# TP : 1을 1이라고 하는 경우 
# FN : 1을 0이라고 하는 경우 
# FP : 0을 1이라고 하는 경우 
# TN : 0을 0이라고 하는 경우 
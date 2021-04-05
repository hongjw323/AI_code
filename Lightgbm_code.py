import numpy as np
import matplotlib.pyplot as plt
import pickle 
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier, plot_importance

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn.model_selection import GridSearchCV
import joblib

# data preprocessing
print("[*] Start : LightGBM Classification")

datasets = np.loadtxt('파일.csv', delimiter=',',skiprows=1)
xy_data = datasets   # exclude header
train_set, test_set = train_test_split(xy_data, test_size=0.4, random_state=123)   
print('Training Length : ', len(train_set), 'Test Length : ', len(test_set))

x_train_data = train_set[:,1:]
y_train_data = train_set[:,:1].reshape(-1,)

x_test_data = test_set[:,1:]
y_test_data = test_set[:,:1].reshape(-1,)

#print("[x_train_data]",x_train_data.shape)
#print("[y_train_data]", y_train_data.shape)
#print("[x_test_data]", x_test_data.shape)
#print("[y_test_data]", y_test_data.shape)

lgb = LGBMClassifier(n_estimators=1500, learning_rate=0.1, max_depth=15, application='binary', num_leaves=30, metrics='binary_logloss')
classifier = lgb.fit(x_train_data, y_train_data)

print(lgb.score(x_train_data, y_train_data))
#print(lgb.score(x_test_data, y_test_data))

y_pred = lgb.predict(x_test_data)
#y_pred = classifier.predict_proba(x_test_data)
#print(y_pred)
# for yy in y_pred:
#   print(yy)
print(confusion_matrix(y_test_data,y_pred))
print(classification_report(y_test_data,y_pred))

fig, ax = plt.subplots(figsize=(10,20))
plot_importance(lgb, ax, max_num_features=32)
plt.show()
# model save
#joblib.dump(lgb, open('lgb.model', 'wb'))
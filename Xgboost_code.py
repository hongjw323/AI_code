import numpy as np
import matplotlib.pyplot as plt
import pickle 
from xgboost import XGBClassifier

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn.model_selection import GridSearchCV
import joblib

# data preprocessing
print("[*] Start : XGBoost Classification")

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

# min_child_weight : 과적합 방지목적, 너무 높은 값은 과소적합을 야기하기 때문에 cv를 사용해서 적절하게 제시해야 함 (기본 설정값 1)
# max_depth : 과적합 방지목적, 보통 3 - 10 값이 적용됨 (기본 설정값 6)
# n_estimators : 
xgb_param = {
    'min_child_weight': [1, 5, 10],
    'max_depth': [4,6,8,10],
    'n_estimators':[10,20,30],
}

xgb = XGBClassifier(n_estimators=500, max_depth=8) # best 
#xgb = XGBClassifier(n_estimators=1500, max_depth=20, learning_rate=0.1)
classifier = xgb.fit(x_train_data, y_train_data)

print(xgb.score(x_train_data, y_train_data))
print(xgb.score(x_test_data, y_test_data))

y_pred = xgb.predict(x_test_data)
#y_pred = classifier.predict_proba(x_test_data)
#print(y_pred)
# for yy in y_pred:
#   print(yy)
print(confusion_matrix(y_test_data,y_pred))
print(classification_report(y_test_data,y_pred))

# xgb.plot_importance(xgb)
# xgb.plot_tree(xgb, num_trees=3)

# model save
# joblib.dump(xgb, open('xgboost.model', 'wb'))
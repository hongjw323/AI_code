import numpy as np
import matplotlib.pyplot as plt
import pickle 
from tensorflow.keras.models import load_model
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras import models
from keras.utils import np_utils
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_validate
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier, plot_importance

# data preprocessing
print("[*] Start : Ensemble Classification")

datasets = np.loadtxt('파일.csv', delimiter=',',skiprows=1)
xy_data = datasets   # exclude header
train_set, test_set = train_test_split(xy_data, test_size=0.4, random_state=123)

x_train_data = train_set[:,1:]
y_train_data = train_set[:,:1].reshape(-1,)

x_test_data = test_set[:,1:]
y_test_data = test_set[:,:1].reshape(-1,)

print("[x_train_data]",x_train_data.shape)
print("[y_train_data]", y_train_data.shape)
print("[x_test_data]", x_test_data.shape)
print("[y_test_data]", y_test_data.shape)

rf = RandomForestClassifier(n_estimators=80, random_state=123)
lgb = LGBMClassifier(n_estimators=1500, learning_rate=0.1, max_depth=15, application='binary', num_leaves=30, metrics='binary_logloss')
xgb = XGBClassifier(n_estimators=500, max_depth=8)

models = [
    ('rf', rf),
    ('lgb', lgb),
    ('xgb', xgb)
]

# hard vote
hard_vote  = VotingClassifier(models, voting='hard')
#hard_vote_cv = cross_validate(hard_vote, x_train_data, y_train_data, cv=k_fold)
hard_vote.fit(x_train_data, y_train_data)
pred = hard_vote.predict(x_train_data)

# soft vote
# soft_vote  = VotingClassifier(models, voting='soft')

# soft_vote.fit(x_train_data, y_train_data)
# pred = soft_vote.predict(x_train_data)

print('Voting 분류기 정확도: {0:.4f}'.format(accuracy_score(y_train_data, pred)))

# classifiers = [rf, lgb, xgb]
# for classifier in classifiers:
#     classifier.fit(x_train_data, y_train_data)
#     pred = classifier.predict(x_train_data)
#     class_name= classifier.__class__.__name__
#     print('{0} 정확도: {1:.4f}'.format(class_name, accuracy_score(y_train_data , pred)))
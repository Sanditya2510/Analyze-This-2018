# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 12:45:48 2018

@author: Anshit Vishwakarma
"""

import pandas as pd
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import matplotlib.pyplot as plt
import keras

# sklearn tools for model training and assesment
from sklearn.model_selection import train_test_split
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.metrics import (roc_curve, auc, accuracy_score)

# specify your configurations as a dict
params = {'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_class':7,
    'metric': 'multi_logloss',
    'learning_rate': 0.6,
    'max_bin': 70,
    'num_leaves': 50
#    'feature_fraction': 0.4,
#    'bagging_fraction': 0.6,
#    'bagging_freq': 17
}
# Importing the libraries

dataset=pd.read_csv('train.csv')
dataset = dataset.fillna(dataset.median())
dataset.drop(['Name','Ticket','Cabin'], axis=1, inplace=True)
dataset = pd.get_dummies(dataset, columns=['Sex', 'Embarked'], drop_first=True)


dataset2=pd.read_csv('test.csv')
dataset2 = dataset2.fillna(dataset2.median())
dataset2.drop(['Name','Ticket','Cabin'], axis=1, inplace=True)
dataset2 = pd.get_dummies(dataset2, columns=['Sex', 'Embarked'], drop_first=True)


X2 = dataset2.iloc[:, [1,2,3,4,5,6,7,8]].values
X2 = pd.DataFrame(X2)
#Taking care of missing data
#from sklearn.preprocessing import Imputer
#imputer = Imputer(missing_values = 'NaN', strategy='mean', axis = 0)
#imputer = imputer.fit(dataset.values[:, 0:889])
#dataset.values[:, 0:889] = imputer.transform(dataset.values[:, 0:889])

#dividing the dependent and independent variables
X = dataset.iloc[:, [2,3,4,5,6,7,8,9]].values
X = pd.DataFrame(X)
y = dataset.iloc[:, 1].values
y = pd.DataFrame(y)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

# train
gbm = lgb.train(params,
                lgb_train,
                valid_sets=lgb_eval)

y_pred = gbm.predict(X2)
y_pred = pd.DataFrame(y_pred)
y_pred=y_pred.idxmax(axis=1)
y_test = pd.DataFrame(y_test)
y_pred = pd.DataFrame(y_pred)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_pred, y_test)

import sklearn
sklearn.metrics.accuracy_score(y_test, y_pred)

#y_pred = y_pred + 1
gridParams = {
    'learning_rate': [0.0001,0.001,0.01,0.1],
    'num_leaves': [40,60,80,100],
    'boosting_type' : ['gbdt'],
    'reg_alpha': [0.0001,0.001,0.01,0.1],
    'reg_lambda': [0.0001,0.001,0.01,0.1],
    'max_bin': [40,60,80,100]
}

mdl = lgb.LGBMClassifier(
    task = params['task'],
    metric = params['metric'],
    boosting_type = params['boosting_type'],
    objective = params['objective'],
#    max_bin = params['max_bin'],
    num_class = params['num_class'],
#    feature_fraction = params['feature_fraction'],
#    bagging_fraction = params['bagging_fraction'],
#    bagging_freq = params['bagging_freq'],
#    min_data_in_leaf = params['min_data_in_leaf'],
#    min_sum_hessian_in_leaf = params['min_sum_hessian_in_leaf'],
#    is_enable_sparse = params['is_enable_sparse'],
#    use_two_round_loading = params['use_two_round_loading'],
#    is_save_binary_file = params['is_save_binary_file'],
    n_jobs = -1
)


# Create the grid
grid = GridSearchCV(mdl, gridParams, verbose=2, cv=5, scoring=scoring, n_jobs=-1, refit='AUC')
# Run the grid
grid.fit(X_train, y_train)

print('Best parameters found by grid search are:', grid.best_params_)
print('Best score found by grid search is:', grid.best_score_)
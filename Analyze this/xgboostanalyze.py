# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 10:30:40 2018

@author: sandi
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

dataset = pd.read_csv('Training_dataset_Original.csv')

dataset.replace('missing', np.nan, inplace=True)
dataset.replace('na', np.nan, inplace=True)

dataset = dataset.replace('L',1)
dataset = dataset.replace('C',0)


# from sklearn.preprocessing import Imputer
# imputer = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)
# imputer = imputer.fit(X[:, :47])
# X[:, :47] = imputer.transform(X[:, :47])
# dataset.interpolate(method='linear', inplace=True)
dataset = dataset.apply(lambda x: x.fillna(x.median()),axis=0)

X = dataset.iloc[:, :48].values
y = dataset.iloc[:,48].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# =============================================================================
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X_train_std = sc.fit_transform(X_train)
# X_test_std = sc.fit_transform(X_test)
# 
# =============================================================================
#seeing the eigen values(PCA)
# =============================================================================
# cov_mat = np.cov(X_train_std.T)
# eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
# print('\nEigenValues\n%s' %eigen_vals)
# 
# tot = sum(eigen_vals)
# var_exp = [(i/tot) for i in sorted(eigen_vals, reverse=True)]
# cum_var_exp = np.cumsum(var_exp)
# =============================================================================

# =============================================================================
# import matplotlib.pyplot as plt
# plt.bar(range(1,49), var_exp, alpha=0.5, align='center',label='individual explained variance')
# plt.step(range(1,49), cum_var_exp, where='mid',label='cumulative explained variance')
# plt.ylabel('Explained variance ratio')
# plt.xlabel('Principal components')
# plt.legend(loc='best')
# plt.show()
# =============================================================================

#Stratified K-fold
# =============================================================================
# from sklearn.model_selection import StratifiedKFold
# kfold = StratifiedKFold(
#                         n_splits=10
#                         )
# for train_index, test_index in kfold.split(X,y): 
#     print("Train:", train_index, "Validation:", test_index) 
#     X_train, X_test = X[train_index], X[test_index] 
#     y_train, y_test = y[train_index], y[test_index]
# 
# =============================================================================







from xgboost import XGBClassifier
classifier = XGBClassifier(n_estimators=600,
                   
                   max_depth=7,
                   learning_rate=0.05,
                   subsample=0.8,
                   colsample_bytree=0.4,
                   gamma=0.05 
                   )
classifier.fit(X_train, y_train)

y_pred = classifier.predict_proba(X_train)
roc_auc_score(y_train, y_pred[:,1])

y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

(cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])

# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{
               'max_depth':[4,5,6,7,8],
               'learning_rate':[0.03,0.001,0.005,0.007,0.05],
               'n_estimators':[100,200,300,400,500,600,700,800]              
               }]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

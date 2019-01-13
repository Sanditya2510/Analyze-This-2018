# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 07:39:43 2018

@author: Anshit Vishwakarma
"""





import pandas as pd
import numpy as np
dataset=pd.read_csv('Training_dataset_Original.csv')
dataset = dataset.fillna(dataset.median())
dataset=dataset.replace(['C','L'],[0,1])

X = dataset.iloc[:, 1:48].values
y = dataset.iloc[:, 48].values

leadset=pd.read_csv('Leaderboard_dataset.csv')
leadset = leadset.fillna(leadset.median())
leadset=leadset.replace(['C','L'],[0,1])
X2 = leadset.iloc[:, 1:48].values

'''# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]'''

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting XGBoost to the Training set
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()

# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'reg_alpha': [0.1,0.2,0.3,0.4,0.5], 'gamma':[0.11,0.12,0.13,0.14,0.15], 'learning_rate':[0.01,0.04,0.08,0.1]}]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
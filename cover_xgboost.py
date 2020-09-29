# -*- coding: utf-8 -*-
"""
Extreme Gradient Boosting Tree Pipeline

@author: Nick
"""


import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


# read in the data
test = pd.read_csv("cover test.csv")
train = pd.read_csv("cover train.csv")

# define the train/test split
train_idx = np.arange(train.shape[0])
test_idx = np.arange(test.shape[0]) + train.shape[0]

# split data in X and Y
Y = train[["Cover_Type"]].copy()
X = pd.concat([train.drop(columns="Cover_Type"), test], axis=0)
del train, test

# set up the pipeline
pipeline = Pipeline([
    ('var', VarianceThreshold()),
    ('model', XGBClassifier(booster="gbtree", 
                            colsample_bytree=0.8,
                            subsample=0.8,
                            random_state=42, n_jobs=-1)),
])

# set up the grid search
parameters = {
    'model__n_estimators': (25, 50, 100),
    'model__learning_rate': (0.001, 0.01, 0.1),
    'model__max_depth': (5, 10),
    'model__min_child_weight': (1, 4),
}
grid_search = GridSearchCV(pipeline, parameters, cv=2, n_jobs=1, verbose=1)

# search the grid for the best the model
grid_search.fit(X.iloc[train_idx, :], Y.iloc[train_idx, :])
model = grid_search.best_estimator_

# predict training and testing data
train_predict = pd.DataFrame(model.predict(X.iloc[train_idx, :]), columns=Y.columns)
test_predict = pd.DataFrame(model.predict(X.iloc[test_idx, :]), columns=Y.columns)

# collect actual data
train_actual = Y.iloc[train_idx, :].copy().reset_index(drop=True)
# test_actual = Y.iloc[test_idx, :].copy().reset_index(drop=True)

# score the goodness of fit
train_score = []
test_score = []
for j in range(train_predict.shape[1]):
    train_score.append(accuracy_score(train_actual.iloc[:,j], train_predict.iloc[:,j]))
    # test_score.append(accuracy_score(test_actual.iloc[:,j], test_predict.iloc[:,j]))
train_score = np.mean(train_score)
# test_score = np.mean(test_score)

# report the scores
print("Train Accuracy: " + str(train_score) + "\n"
      "Test Accuracy: " + str(test_score))

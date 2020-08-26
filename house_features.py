# -*- coding: utf-8 -*-
"""
Creates 2nd order polynomial features
Selects best features using Random Forest

@author: Nick
"""


import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import RFE


# read in the data
data = pd.read_csv("house.csv")

# determine which columns are strings (for X)
x_columns = data.columns
x_dtypes = data.dtypes
x_str = np.where(x_dtypes == "object")[0]

# convert any string columns to binary columns
data = pd.get_dummies(data, columns=x_columns[x_str])

# fill in missing values with the (column) average
impute = SimpleImputer(strategy="mean")
columns = data.columns
data = pd.DataFrame(data=impute.fit_transform(data), columns=columns)

# separate X and Y
X = data.drop(columns=["Id", "SalePrice"]).copy()
Y = data.drop(columns=X.columns).drop(columns="Id").copy()

# determine if we are building a classifier model
classifier = np.all(np.unique(Y.to_numpy()) == [0, 1])
outputs = Y.shape[1]

# add 2nd order polynomial features to X
poly = PolynomialFeatures(2, include_bias=False)
x_columns = X.columns
X = pd.DataFrame(poly.fit_transform(X))
X.columns = poly.get_feature_names(x_columns)

# separate the data into training and testing
np.random.seed(1)
test_idx = np.random.choice(a=X.index.values, size=int(X.shape[0] / 5), replace=False)
train_idx = np.array(list(set(X.index.values) - set(test_idx)))

# set up the model
if classifier:
    selector = RFE(RandomForestClassifier(n_estimators=25,
                                          max_depth=10,
                                          min_samples_leaf=1,
                                          max_features="sqrt",
                                          random_state=42,
                                          n_jobs=-1), step=0.05, verbose=1)
else:
    selector = RFE(RandomForestRegressor(n_estimators=25,
                                         max_depth=10,
                                         min_samples_leaf=1,
                                         max_features="sqrt",
                                         random_state=42,
                                         n_jobs=-1), step=0.05, verbose=1)

# determine which features to keep
C = X.shape[1] # columns
R = X.shape[0] # rows
while C > R / 5:
    keep_idx = np.repeat(0, X.shape[1])
    for j in Y.columns:
        selector.fit(X, Y.loc[:, j])
        keep_j = selector.support_ * 1
        keep_idx = keep_idx + keep_j
        print("--")
    keep = np.where(keep_idx > 0)[0]
    X = X.iloc[:, keep]
    C = X.shape[1] # columns
    R = X.shape[0] # rows

# export the data
X.to_csv("X house.csv", index=False)
Y.to_csv("Y house.csv", index=False)

# -*- coding: utf-8 -*-
"""
Creates 2nd order polynomial features
Selects unique features using Hierarchical Clustering

@author: Nick
"""


import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cluster import FeatureAgglomeration


# read in the data
data = pd.read_csv("titanic.csv")

# get the text columns form data
text = data[["Name", "Ticket"]]
data = data.drop(columns=["Name", "Ticket"])

# collect the words (and their inverse frequencies) from each document
# 'matrix' is a term (columns) document (rows) matrix
matrix = pd.DataFrame()
for c in text.columns:
    vector = TfidfVectorizer()
    matrix2 = vector.fit_transform(text[c].tolist())
    matrix2 = pd.DataFrame(matrix2.toarray(), columns=vector.get_feature_names())
    matrix = pd.concat([matrix, matrix2], axis=1)

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
X = data.drop(columns=["PassengerId", "Survived"]).copy()
Y = data.drop(columns=X.columns).drop(columns="PassengerId").copy()
X_copy = X.copy()

# add 2nd order polynomial features to X
poly = PolynomialFeatures(2, include_bias=False)
x_columns = X_copy.columns
X_copy = pd.DataFrame(poly.fit_transform(X_copy))
X_copy.columns = poly.get_feature_names(x_columns)

# add the term-document matrix to X
X_copy = pd.concat([X_copy, matrix], axis=1)

# drop any constant columns in X
X_copy = X_copy.loc[:, (X_copy != X_copy.iloc[0]).any()]

# standardize the data to take on values between 0 and 1
X = ((X_copy - X_copy.min()) / (X_copy.max() - X_copy.min())).copy()

# build the feature selection model
num = 222
hclust = FeatureAgglomeration(n_clusters=num, linkage="ward", distance_threshold=None)
hclust.fit(X)

# collect the features to keep
clusters = hclust.labels_
keep = []
for i in range(num):
    keep.append(np.where(clusters == i)[0][0])
X = X_copy.iloc[:, keep]

# export the data
X.to_csv("X titanic.csv", index=False)
Y.to_csv("Y titanic.csv", index=False)

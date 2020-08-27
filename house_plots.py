# -*- coding: utf-8 -*-
"""
https://plotly.com/python/plotly-express/

@author: Nick
"""

import pandas as pd
import plotly.express as px
from plotly.offline import plot

df = pd.read_csv("house.csv").drop(columns="Id")

# define the target feature and other meaningful features
targets = ["SalePrice"]
features = ["OverallQual", "YearBuilt", "GrLivArea", "GarageCars", "GarageArea", 
            "TotalBsmtSF", "PavedDrive", "BldgType"]

# plot a matrix of scatter plots to see the whole data set
fig = px.scatter_matrix(df, 
                        dimensions=["SalePrice", "OverallQual", "YearBuilt", 
                                    "GrLivArea", "GarageCars", "GarageArea", 
                                    "TotalBsmtSF"],
                        color="PavedDrive",
                        opacity=0.7)
fig.update_traces(diagonal_visible=False)
plot(fig)

# these 3 variables have the strongest linear relationship with eachother
group = ["SalePrice", "GrLivArea", "TotalBsmtSF"]

# plot the group of 3 variables across categories
fig = px.scatter_3d(df, x="SalePrice", y="GrLivArea", z="TotalBsmtSF",
                    color="PavedDrive", opacity=0.7)
plot(fig)

# plot two variables across categories
fig = px.density_contour(df, x="SalePrice", y="TotalBsmtSF", marginal_x="histogram", 
                         marginal_y="box", color="PavedDrive")
plot(fig)

# plot SalePrice across categories
fig = px.strip(df, y="SalePrice", color="BldgType")
plot(fig)

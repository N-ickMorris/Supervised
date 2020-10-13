# -*- coding: utf-8 -*-
"""
Collects NFL Team Performance Data

@author: Nick
"""


import numpy as np
import pandas as pd
from sportsreference.nfl.teams import Teams
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LassoCV
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot


# what years should we evaluate?
start_year = 1998
end_year = 2018

# query for the data
nfl_ = pd.DataFrame()
for y in range(start_year, end_year + 1):
    teams = Teams(year=y)
    print("---- " + str(y) + " ----")
    for t in teams:
        print(t.name)
        df = t.schedule.dataframe
        df["team"] = t.name
        nfl_ = pd.concat([nfl_, df], axis=0)    

# remove missing values
nfl_ = nfl_.dropna()

# convert time_of_possession into minutes
nfl_["time_of_possession"] = "00:" + nfl_["time_of_possession"]
nfl_["time_of_possession"] = pd.to_datetime(nfl_["time_of_possession"])
minutes = nfl_["time_of_possession"].dt.minute
minutes = minutes + (nfl_["time_of_possession"].dt.second / 60)
nfl_["time_of_possession"] = minutes

# create a team/year variable
nfl_["datetime"] = pd.to_datetime(nfl_["datetime"])
nfl_["year"] = nfl_["datetime"].dt.year.astype(str)
nfl_["team_year"] = nfl_["team"] + nfl_["year"]

# convert result into a binary variable
Y = pd.get_dummies(nfl_[["result"]])[["result_Win"]]
Y["team_year"] = nfl_["team_year"]

# get features (X)
X = nfl_.drop(columns=['boxscore_index', 'date', 'datetime', 'day',
                       'opponent_name', 'overtime', 'result', 'type',
                       'week', 'team', 'year', 'location', 'opponent_abbr'])

# compute wins per team
wins = Y.groupby("team_year").mean()

# compute performance statistics per team
means = X.groupby("team_year").mean()
stds = X.groupby("team_year").std()

# update columns names
means.columns = [c + " mean" for c in means.columns]
stds.columns = [c + " sd" for c in stds.columns]

# update X and Y
Y = wins.copy().reset_index(drop=True)
X = pd.concat([means, stds], axis=1).reset_index(drop=True)

# fill in missing values
X = X.fillna(method="bfill").fillna(method="ffill")

# split the data into training and testing
np.random.seed(1)
test_idx = np.random.choice(a=X.index.values, size=int(X.shape[0] / 5), replace=False)
train_idx = np.array(list(set(X.index.values) - set(test_idx)))

# set up a machine learning pipeline
pipeline = Pipeline([
    ('var', VarianceThreshold()),
    ('scale', MinMaxScaler()),
    # ('model', LassoCV(eps=1e-9, n_alphas=16, n_jobs=-1)),
    # ('model', BayesianRidge()),
    ('model', RandomForestRegressor(n_estimators=50, max_depth=8, min_samples_leaf=1, n_jobs=-1, random_state=42)),
    # ('model', MLPRegressor(max_iter=200, hidden_layer_sizes=(128, 128), learning_rate_init=0.001, batch_size=32, activation="relu", solver="adam", learning_rate="adaptive", random_state=42)),
])

# train the model
pipeline.fit(X.iloc[train_idx, :], Y.iloc[train_idx, :])

# predict the test set
predict = pipeline.predict(X.iloc[test_idx, :])
actual = Y.iloc[test_idx, :].to_numpy()[:,0]

# score the forecast
print("R2: " + str(r2_score(actual, predict)))

# prepare the data for plotting
df = pd.DataFrame({"Predict": predict, "Actual": actual})
df["index"] = means.iloc[test_idx, :].index

# plot the prediction series
fig = px.line(df, x="index", y="Predict")
fig.add_trace(go.Scatter(x=df["index"], y=df["Actual"], mode="lines", showlegend=False, name="Actual"))
fig.update_layout(font=dict(size=16))
plot(fig, filename="Series Predictions.html")

# draw a parity plot
fig1 = px.scatter(df, x="Actual", y="Predict")
fig1.add_trace(go.Scatter(x=df["Actual"], y=df["Actual"], mode="lines", showlegend=False, name="Actual"))
fig1.update_layout(font=dict(size=16))
plot(fig1, filename="Parity Plot.html")



# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 10:31:22 2021

@author: Julian
"""

# evaluate an xgboost regression model on the housing dataset
import plotly.io as pio
import plotly.graph_objects as go
from numpy import absolute
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from numpy import loadtxt
from xgboost import plot_tree
import matplotlib.pyplot as plt

df = pd.read_csv("304L_WeldData_April2021.csv")
df = df.drop(df.index[[49, 50, 51, 52, 53, 54, 54]])

# assign targets and labels
xdata = df[['Power \n(W)', 'Travel \nSpeed\n(mm/sec)']].values
ydata = df[['Weld \nDepth\n(mm)']].values

X_train, X_test, y_train, y_test = train_test_split(
    xdata, ydata, test_size=0.30)

# param_grid = {'n_estimators' : [100,200,400,600,1000,1500,2000],
#               'max_depth' : [1,2,3,4,5,6,7,8,9,10],
#               'eta' : [0.001,0.01,0.1,0.2,0.3],
#               'subsample' : [0.2,0.4,0.6,0.8,1],
#               'colsample_bytree' : [0.2,0.4,0.6,0.8,1]}

# grid = GridSearchCV(XGBRegressor(), param_grid, refit = True, verbose = 3, n_jobs = -1)

# # fitting the model for grid search
# grid.fit(X_train, y_train)

# # print best parameter after tuning
# print(grid.best_params_)
# grid_predictions = grid.predict(X_test)


model = XGBRegressor(n_estimators=1000, max_depth=2,
                     eta=0.2, subsample=0.2, colsample_bytree=1)
# define model evaluation method
cv = RepeatedKFold(n_splits=10, n_repeats=3)
# evaluate model
scores = cross_val_score(model, X_train, y_train,
                         scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# force scores to be positive
scores = absolute(scores)
print('Mean MAE: %.3f (%.3f)' % (scores.mean(), scores.std()))

model.fit(X_train, y_train)

yhat = model.predict(X_test)

pio.renderers.default = 'browser'

table = go.Figure(data=[go.Table(header=dict(values=['Test Values', 'Predicted Values']),
                                 cells=dict(values=[y_test, yhat]))
                        ])
table.show()

print("MSE : ", mse(y_test, yhat))
print("MAE : ", mae(y_test, yhat))

plot_tree(model, num_trees=20)
plt.show()
plot_tree(model, num_trees=20)
plt.show()

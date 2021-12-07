# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 10:31:22 2021

@author: Julian
"""

# evaluate an xgboost regression model

import plotly.io as pio
import plotly.graph_objects as go
import numpy as np
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
from mpl_toolkits.mplot3d import Axes3D

#Dataset interpolated using a midpoint interpolation
df = pd.read_csv("interpolated_data.csv")
#df = pd.read_csv("NN_Development.csv")

#uncomment to plot features and target
# threeplot = plt.figure().add_subplot(projection='3d')
# threeplot.scatter(df['Power \n(W)'],
#                   df['Travel \nSpeed\n(mm/sec)'], df['Weld \nDepth\n(mm)'])
# threeplot.set_xlabel('Power \n(W)')
# threeplot.set_ylabel('Travel \nSpeed\n(mm/sec)')
# threeplot.set_zlabel('Weld \nDepth\n(mm)')
# plt.show()

# assign targets and labels
xdata = df[['Power \n(W)', 'Travel \nSpeed\n(mm/sec)']].values
ydata = df[['Weld \nDepth\n(mm)']].values

X_train, X_test, y_train, y_test = train_test_split(
    xdata, ydata, test_size=0.30)

#grid search parameters
param_grid = {'n_estimators': [1000, 1300, 1500, 1600, 1800, 2000],
              'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10],
              'eta': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3],
              'subsample': [0.2, 0.4, 0.5, 0.6, 0.8, 1],
              'colsample_bytree': [0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 1]}

grid = GridSearchCV(XGBRegressor(), param_grid,
                    refit=True, verbose=3, n_jobs=-1)

# fitting the model for grid search
grid.fit(X_train, y_train)

# print best parameter after tuning
print(grid.best_params_)
grid_predictions = grid.predict(X_test)



# model = XGBRegressor(n_estimators=1600, max_depth=8, eta=0.1,
#                      subsample=0.2, colsample_bytree=1)
# # define model evaluation method
# cv = RepeatedKFold(n_splits=10, n_repeats=3)
# # evaluate model
# scores = cross_val_score(model, X_train, y_train,
#                          scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# # force scores to be positive
# scores = absolute(scores)
# print('Mean MAE: %.3f (%.3f)' % (scores.mean(), scores.std()))

# model.fit(X_train, y_train)

##return predictions
# yhat = model.predict(X_test)

# pio.renderers.default = 'browser'

# # Plot Data
# data = np.transpose(
#     np.vstack((np.transpose(y_test), np.around(yhat, decimals=3))))
# finaldf = pd.DataFrame(data=data, columns=["Actual", "Predicted"])
# #finaldf.to_csv('xgb_predictions_graph.csv', index=False)

# # table = go.Figure(data=[go.Table(header=dict(values=['Test Values', 'Predicted Values']),
# #                                  cells=dict(values=[y_test, yhat]))
# #                         ])
# # table.show()

# print("MSE : ", mse(y_test, yhat))
# print("MAE : ", mae(y_test, yhat))
# x = np.arange(len(yhat))
# plt.scatter(x, np.sort(data[:, 0]), color='green')
# plt.scatter(x,  np.sort(data[:, 1]), color='hotpink')
# plt.legend(["Predictions", "Actual Values"])
# plt.xlabel("Sample")
# plt.ylabel("Weld Depth (mm)")
# # plot_tree(model, num_trees=1)
# # plot_tree(model, num_tree=20)
# # plt.show()
# # plt.show()

# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 19:35:50 2021

@author: Julian
"""
import numpy as np
from sklearn.model_selection import KFold

import torch
import torch.nn as nn

import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader

from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from sklearn.model_selection import KFold

import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn import preprocessing

from sklearn.metrics import mean_absolute_error as mae

from skorch import NeuralNetRegressor
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split

nin = 2
nh0 = 11
nh1 = 33
nh2 = 105
nout = 1
# nin: dimension of input data
# nh: number of hidden units
# nout: number of outputs


class optimal_Net(nn.Module):
    def __init__(self, nin, nh0, nh1, nh2, nout):
        super(optimal_Net, self).__init__()
        self.activation = nn.ReLU()
        self.Dense1 = nn.Linear(nin, nh0)
        self.Dense2 = nn.Linear(nh0, nh1)
        self.Dense3 = nn.Linear(nh1, nh2)
        self.Dense4 = nn.Linear(nh2, nout)

    def forward(self, x):
        x = self.activation(self.Dense1(x))
        x = self.activation(self.Dense2(x))
        x = self.activation(self.Dense3(x))
        out = self.Dense4(x)
        return out


df = pd.read_csv("interpolated_data.csv")
#df = pd.read_csv("NN_Development.csv")

# assign targets and labels
xdata1 = df[['Power \n(W)', 'Travel \nSpeed\n(mm/sec)']
            ].values.astype(np.float32)
ydata = df[['Weld \nDepth\n(mm)']].values.astype(np.float32)

print("Features:", xdata1.shape, "\nTargets:",
      ydata.shape, "\nSamples:", df.shape[0])


# preprocess data min-max scalar
scaler = preprocessing.StandardScaler()
xdata = scaler.fit_transform(xdata1)

model = optimal_Net(nin, nh0, nh1, nh2, nout)

#cv = RepeatedKFold(n_splits=100, n_repeats=3)

regression = NeuralNetRegressor(model, max_epochs=200, optimizer=optim.Adam,
                                optimizer__lr=0.01493647503759509, optimizer__weight_decay=1.9103479937590043e-05, batch_size=30)

# scores = cross_val_score(regression, xdata, ydata, cv=cv,
# scoring="neg_mean_absolute_error")
#scores = np.absolute(scores)
#print('Mean MAE: %.3f (%.3f)' % (scores.mean(), scores.std()))
# y_pred = cross_val_predict(regression, xdata, ydata, cv=5)

# print(mae(ydata, y_pred))
X_train, X_test, y_train, y_test = train_test_split(
    xdata, ydata, test_size=0.30)
regression.fit(X_train, y_train)

yhat = regression.predict(X_test)
print(mae(y_test, yhat))

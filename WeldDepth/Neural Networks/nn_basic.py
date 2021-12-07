# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 15:48:52 2021

@author: Julian
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 19:35:50 2021

@author: Julian
"""


from sklearn.model_selection import train_test_split
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
nin = 2
nh0 = 12
nh1 = 10
nout = 1
# nin: dimension of input data
# nh: number of hidden units
# nout: number of outputs


class Net(nn.Module):
    def __init__(self, nin, nh0, nh1, nout):
        super(Net, self).__init__()
        self.activation = nn.ReLU()
        self.Dense1 = nn.Linear(nin, nh0)
        self.Dense2 = nn.Linear(nh0, nh1)
        self.Dense3 = nn.Linear(nh1, nout)

    def forward(self, x):
        x = self.activation(self.Dense1(x))
        x = self.activation(self.Dense2(x))
        out = self.Dense3(x)
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

model = Net(nin, nh0, nh1, nout)

cv = RepeatedKFold(n_splits=10, n_repeats=3)

regression2 = NeuralNetRegressor(model, max_epochs=200, optimizer=optim.Adam,
                                 optimizer__lr=0.01,
                                 batch_size=30)

scores = cross_val_score(regression2, xdata, ydata, cv=cv,
                         scoring="neg_mean_absolute_error")
scores = np.absolute(scores)
print('Mean MAE: %.3f (%.3f)' % (scores.mean(), scores.std()))
# y_pred = cross_val_predict(regression, xdata, ydata, cv=5)

# print(mae(ydata, y_pred))
X_train, X_test, y_train, y_test = train_test_split(
    xdata, ydata, test_size=0.30)
regression2.fit(X_train, y_train)

yhat = regression2.predict(X_test)
print(mae(y_test, yhat))

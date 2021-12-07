# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 07:23:42 2021

@author: Julian
"""

import plotly.io as pio
import plotly.graph_objects as go
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn import model_selection
from sklearn import preprocessing
import torch.nn as nn
import torch.optim
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae

df = pd.read_csv("interpolated_data.csv")

# assign targets and labels
xdata1 = df[['Power \n(W)', 'Travel \nSpeed\n(mm/sec)']].values
ydata = df[['Weld \nDepth\n(mm)']].values

print("Features:", xdata1.shape[1], "\nTargets:",
      ydata.shape[1], "\nSamples:", df.shape[0])


# preprocess data min-max scalar
# scaler = preprocessing.StandardScaler()
# xdata = scaler.fit_transform(xdata1)

scaler = preprocessing.StandardScaler()
xdata = scaler.fit_transform(xdata1)


# Turn data into tensors


# training test split for dataset
new_df = np.concatenate((xdata, ydata), axis=1)
train_data, test_data = model_selection.train_test_split(new_df, test_size=0.3)

# splitting data into targets and features
x_tr = scaler.fit_transform(train_data[:, [0, 1]])
y_tr = train_data[:, [2]]

x_ts = scaler.fit_transform(test_data[:, [0, 1]])
y_ts = test_data[:, [2]]

# putting data into tensor form
xtr_torch = torch.Tensor(x_tr)
ytr_torch = torch.Tensor(y_tr)

xts_torch = torch.Tensor(x_ts)

# inputting data into dataloader
train_dataset = TensorDataset(xtr_torch, ytr_torch)
train_loader = DataLoader(train_dataset, batch_size=30, shuffle=True)

#
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = torch.nn.MSELoss()
num_epoch = 200

nin = 2
nh0 = 12
nh1 = 10
nout = 1
# nin: dimension of input data
# nh: number of hidden units
# nout: number of outputs


class optimal_Net(nn.Module):
    def __init__(self, nin, nh0, nh1, nout):
        super(optimal_Net, self).__init__()
        self.activation = nn.ReLU()
        self.Dense1 = nn.Linear(nin, nh0)
        self.Dense2 = nn.Linear(nh0, nh1)
        self.Dense3 = nn.Linear(nh1, nout)

    def forward(self, x):
        x = self.activation(self.Dense1(x))
        x = self.activation(self.Dense2(x))
        out = self.Dense3(x)
        return out


model2 = optimal_Net(nin=nin, nh0=nh0, nh1=nh1, nout=nout)

print(str(model2))


opt = torch.optim.Adam(model2.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()  # this is for regression mean squared loss

#

print_intvl = 50
a_loss = np.zeros([num_epoch])

# Outer loop over epochs
for epoch in range(num_epoch):
    batch_loss = []
    # Inner loop over mini-batches
    for batch, data in enumerate(train_loader):
        x_batch, y_batch = data
        y_batch = y_batch.view(-1, 1)  # resizes y_batch to (batch_size,1)
        out = model2(x_batch)
        # Compute loss
        loss = criterion(out, y_batch)
        batch_loss.append(loss.item())
        # Compute gradients using back-propagation
        opt.zero_grad()
        loss.backward()
        # Take an optimization 'step' (i.e., update parameters)
        opt.step()

    a_loss[epoch] = np.mean(batch_loss)  # Compute average loss over epoch

    # Print details if epoch is multiple of print_intvl
    if (epoch+1) % print_intvl == 0:
        print('Epoch: {0:d}   Loss: {1:e}'.format(epoch+1, a_loss[epoch]))


#

epoch_it = np.arange(1, num_epoch+1)
plt.plot(epoch_it, a_loss)
plt.grid()
plt.xlabel('Epoch')
plt.ylabel('Loss')


def trunc(values, decs=0):
    return np.trunc(values*10**decs)/(10**decs)


with torch.no_grad():
    yhat = model2(xts_torch).detach().numpy()
    rounded = trunc(yhat, decs=2)

pio.renderers.default = 'browser'

table = go.Figure(data=[go.Table(header=dict(values=['Test Values', 'Predicted Values']),
                                 cells=dict(values=[y_ts, rounded]))
                        ])
table.show()

print("MSE : ", mse(y_ts, yhat))
print("MAE : ", mae(y_ts, yhat))

model = nn.Sequential()
model.add_module('W0', nn.Linear(nin, nh0))
model.add_module('ReLU', nn.ReLU())
model.add_module('W1', nn.Linear(nh0, nout))

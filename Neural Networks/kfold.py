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


def reset_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()


def train(fold, model, device, train_loader, optimizer, epoch):
    criterion = torch.nn.MSELoss()  # this is for regression mean squared loss
    model.train()
    for batch_idx, data in enumerate(train_loader):
        x_batch, y_batch = data
        y_batch = y_batch.view(-1, 1)  # resizes y_batch to (batch_size,1)
        optimizer.zero_grad()
        output = model(x_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        if batch_idx % 200 == 0:
            print('Train Fold/Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                fold, epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(fold, model, device, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data, target
            yhat = model(data).detach().numpy()
            # sum up batch loss
            test_loss += mae(target, yhat)

    test_loss /= len(test_loader.dataset)

    print('\nTest set for fold {}: MAE loss: {:.4f}'.format(
        fold, test_loss))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

df = pd.read_csv("interpolated_data.csv")
#df = pd.read_csv("NN_Development.csv")

# assign targets and labels
xdata1 = df[['Power \n(W)', 'Travel \nSpeed\n(mm/sec)']].values
ydata = df[['Weld \nDepth\n(mm)']].values

print("Features:", xdata1.shape[1], "\nTargets:",
      ydata.shape[1], "\nSamples:", df.shape[0])


# preprocess data min-max scalar
scaler = preprocessing.StandardScaler()
xdata = scaler.fit_transform(xdata1)

dataset = np.hstack([xdata, ydata])

model = optimal_Net(nin, nh0, nh1, nh2, nout)
model.apply(reset_weights)

batch_size = 32
folds = 5
epochs = 200

kfold = KFold(n_splits=folds, shuffle=True)

optimizer = optim.Adam(model.parameters(
), lr=0.01493647503759509, weight_decay=1.9103479937590043e-05)
criterion = torch.nn.MSELoss()  # this is for regression mean squared loss
a_loss = np.zeros([epochs])

for fold, (train_idx, test_idx) in enumerate(kfold.split(dataset)):
    print('------------fold no---------{}----------------------'.format(fold))

    X_train, X_test = xdata[train_idx], xdata[test_idx]
    y_train, y_test = ydata[train_idx], ydata[test_idx]

    train_ds = torch.utils.data.TensorDataset(
        torch.Tensor(X_train), torch.Tensor(y_train))
    test_ds = torch.utils.data.TensorDataset(
        torch.Tensor(X_test), torch.Tensor(y_test))
    trainloader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size)
    testloader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=batch_size)

    model.apply(reset_weights)

    for epoch in range(1, epochs + 1):
        train(fold, model, device, trainloader, optimizer, epoch)
        test(fold, model, device, testloader)

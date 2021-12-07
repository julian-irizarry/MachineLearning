# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 07:23:42 2021

@author: Julian
"""

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

df = pd.read_csv("304L_WeldData_April2021.csv")
df = df.drop(df.index[[49,50,51,52,53,54,54]])

#assign targets and labels
xdata1 = df[['Power \n(W)','Travel \nSpeed\n(mm/sec)']].values
ydata = df[['Weld \nDepth\n(mm)']].values

print("Features:",xdata1.shape[1],"\nTargets:",ydata.shape[1],"\nSamples:",df.shape[0])


#preprocess data min-max scalar
# scaler = preprocessing.StandardScaler()
# xdata = scaler.fit_transform(xdata1)

scaler = preprocessing.MinMaxScaler()
xdata = scaler.fit_transform(xdata1)


#Turn data into tensors
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader


#training test split for dataset
new_df = np.concatenate((xdata, ydata), axis = 1)
train_data, test_data = model_selection.train_test_split(new_df, test_size = 0.3)

#splitting data into targets and features
x_tr = scaler.fit_transform(train_data[:,[0,1]])
y_tr = train_data[:,[2]]

x_ts = scaler.fit_transform(test_data[:,[0,1]])
y_ts = test_data[:,[2]]

#putting data into tensor form
xtr_torch = torch.Tensor(x_tr)
ytr_torch = torch.Tensor(y_tr)

xts_torch = torch.Tensor(x_ts)

#inputting data into dataloader
train_dataset = TensorDataset(xtr_torch,ytr_torch)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

#
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def RMSELoss(yhat,y):
    return torch.sqrt(torch.mean((yhat-y)**2))

criterion = RMSELoss

num_epoch = 200

nin = 2
nh0 = 69
nh1 = 51
nh2 = 62
nout = 1
# nin: dimension of input data
# nh: number of hidden units
# nout: number of outputs

class optimal_Net(nn.Module):
    def __init__(self,nin,nh0,nh1,nh2,nout):
        super(optimal_Net,self).__init__()
        self.activation = nn.LeakyReLU()
        self.Dense1 = nn.Linear(nin,nh0)
        self.Dense2 = nn.Linear(nh0,nh1)
        self.Dense3 = nn.Linear(nh1,nh2)
        self.Dense4 = nn.Linear(nh2,nout)
        
    def forward(self,x):
        x = self.activation(self.Dense1(x))
        x = self.activation(self.Dense2(x))
        x = self.activation(self.Dense3(x))
        out = self.Dense4(x)
        return out

model2 = optimal_Net(nin=nin,nh0=nh0,nh1=nh1,nh2=nh2,nout=nout)

print(str(model2))



opt = torch.optim.Adam(model2.parameters(), lr=0.006859934885280088, weight_decay=1.3085078070061801e-05)
criterion = torch.nn.MSELoss()  # this is for regression mean squared loss

#

print_intvl = 50
a_loss = np.zeros([num_epoch])

# Outer loop over epochs
for epoch in range(num_epoch):
    batch_loss = []
    # Inner loop over mini-batches
    for batch, data in enumerate(train_loader):
        x_batch,y_batch = data
        y_batch = y_batch.view(-1,1) # resizes y_batch to (batch_size,1)
        out = model2(x_batch)
        # Compute loss
        loss = criterion(out,y_batch)        
        batch_loss.append(loss.item())
        # Compute gradients using back-propagation
        opt.zero_grad()
        loss.backward()
        # Take an optimization 'step' (i.e., update parameters)
        opt.step()
        
    a_loss[epoch] = np.mean(batch_loss) # Compute average loss over epoch
    
    # Print details if epoch is multiple of print_intvl
    if (epoch+1) % print_intvl == 0:
        print('Epoch: {0:d}   Loss: {1:e}'.format(epoch+1, a_loss[epoch]))
        

#

epoch_it = np.arange(1,num_epoch+1)
plt.plot(epoch_it, a_loss)
plt.grid()
plt.xlabel('Epoch')
plt.ylabel('Loss')

import numpy as np
def trunc(values, decs=0):
    return np.trunc(values*10**decs)/(10**decs)

with torch.no_grad():
    yhat = model2(xts_torch).detach().numpy()
    rounded = trunc(yhat, decs = 2)
    
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = 'browser'

table = go.Figure(data=[go.Table(header=dict(values=['Test Values', 'Predicted Values']),
                 cells=dict(values=[y_ts,rounded]))
                     ])
table.show()

print("RMSE : ", mse(y_ts,yhat,squared=False))
print("MAE : ", mae(y_ts,yhat))
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 13:05:03 2021

@author: Julian
"""

import numpy as np
import pandas as pd
import torch
import optuna
from sklearn.metrics import mean_squared_error as MSE
from sklearn import preprocessing
import torch.nn as nn
import torch.optim as optim

df = pd.read_csv("304L_WeldData_April2021.csv")
df = df.drop(df.index[[49,50,51,52,53,54,54]])

#assign targets and labels
xdata1 = df[['Power \n(W)','Travel \nSpeed\n(mm/sec)']].values
ydata = df[['Weld \nDepth\n(mm)']].values

print("Features:",xdata1.shape[1],"\nTargets:",ydata.shape[1],"\nSamples:",df.shape[0])


#preprocess data min-max scalar
scaler = preprocessing.StandardScaler()
xdata = scaler.fit_transform(xdata1)

#min_max_scaler = preprocessing.MinMaxScaler()
#xdata = min_max_scaler.fit_transform(xdata1)


#Turn data into tensors
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

x_torch = torch.Tensor(xdata)
y_torch = torch.Tensor(ydata)


#Create training and test splits for data
dataset = TensorDataset(x_torch,y_torch)

train_size = int(0.7 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

batch_size = 16

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

#
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = torch.nn.MSELoss()
epochs = 200

def define_model(trial):
    n_layers = trial.suggest_int("n_layers", 1, 10)
    layers = []
    
    nin = 2
    for i in range(n_layers):
        nh = trial.suggest_int("n_units_1{}".format(i), 4, 128)
        layers.append(nn.Linear(nin, nh))
        layers.append(nn.LeakyReLU())
        
        nin = nh
        
    layers.append(nn.Linear(nin, 1))
    
    return nn.Sequential(*layers)


def objective(trial):
    
    #define the model
    model = define_model(trial).to(device)
    #optimizers to choose from
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    #train the model
    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.view(-1, 1).to(device)
            output = model(data)
            
            #loss
            optimizer.zero_grad()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
        #validation
        batch_MSE = []
        target_mse = []
        output_mse = []
        model.eval()
        with torch.no_grad():
             for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(device), target.view(-1, 1).to(device)
                output = model(data)
                #To calculate MSE might not need to calc MSE here
                target_mse.append(target.cpu().detach().numpy())
                output_mse.append(output.cpu().detach().numpy())
    
                if np.isnan(output_mse[batch_idx]).any() or np.isinf(output_mse[batch_idx]).any():
                    output_mse = np.nan_to_num(output_mse)
                
                batch_MSE.append(MSE(target_mse[batch_idx], output_mse[batch_idx]))
                   
                    
        avg_MSE = np.mean(batch_MSE)
        
        trial.report(avg_MSE, epoch)
        
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
    return avg_MSE


#RUN Optuna
if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100, timeout=600)

    pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
        

optuna.visualization.plot_parallel_coordinate(study)
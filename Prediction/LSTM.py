#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 10:16:44 2021
@author: Usuario
"""

#%% IMPORTS

import numpy as np
import pandas as pd 
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
import math
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy import signal
import matplotlib.pyplot as plt

import random
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.autograd as autograd

import sklearn.metrics
import seaborn as sns
import random
import time
import FCNs_Validation as valFcn
import FCNs_load_data_robolabo as loadFCN
import FCNs_git as gitFCN
from pathlib import Path
import itertools
import pathlib
#%% DATA LOADING
# Número de segmentos totales de la base de datos

# mypath = 'C:\\Users\\Alejandro\\Documents\\MATLAB\\EXTEND\\Tremor_machine_learning\\Proyecto\\'
mypath = '/home/victorl/Tremor_machine_learning/Proyecto'
# path_model_out = 'C:\\Users\\Alejandro\\Documents\\MATLAB\\EXTEND\\Tremor_machine_learning\\Proyecto\\LSTM_Prediction_Alex\\'

data_base= '/BASE_CLINICA_PREDICCION_FILTRADA/'
output_path = '/LSTM_Prediction_Alex/'

# data_base= 'BASE_CLINICA_PREDICCION_FILTRADA\\'
# output_path = 'LSTM_Prediction_Alex\\' 

# CSV file containing the network parameters
model_parameters_file = 'model_parameters_v02.csv'
model_parameters_path = mypath + output_path + model_parameters_file

# ====== DATABASE PARAMETERS ======
num_bloques = 7520
subset_size = 7000
normalize_method = []
random_seed = 100
# =====================

# Extract parameters from CSV
model_parameters = pd.read_csv(model_parameters_path, sep=';')

learning_rate = np.array(model_parameters['LR'])

prediction_horizon = np.array(model_parameters['PH'])

train_window = np.array(model_parameters['SAMP_TRAIN'])

hidden_size = np.array(model_parameters['N_HIDDEN'],dtype=int)

layer_size = np.array(model_parameters['N_LAYERS'])

lstm_type = np.array(model_parameters['LSTM_TYPE'])

# Create model name
model_name = np.empty(0,str)
for k in range(0,len(learning_rate)):
    model_name = np.append(model_name, str(lstm_type[k]) +'_'+ str(layer_size[k]) +'_'+ str(hidden_size[k])
                        +'_'+ str(train_window[k]) +'_'+ str(prediction_horizon[k])
                        +'_'+ str(learning_rate[k]))


n_models = 90
model_ini = 0

# ====== LSTM PARAMETERS ======
batch_size = 10
input_size = 1
output_size = prediction_horizon      # possible choices
# =====================

# ====== TRAINING PARAMETERS ======
n_epochs = 600
# Stop training after N iterations withour decreasing learning rate
stop_treining_steps = 40
# =====================


#%%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# path_script = os.path.dirname(__file__)
# path_project = os.path.dirname(path_script)
path_data_base_in = '/home/victorl/Tremor_machine_learning/Proyecto' + data_base
path_model_out = '/home/victorl/Tremor_machine_learning/Proyecto/' + output_path
# path_data_base_in = Path.home() + '/Tremor_machine_learning/Proyecto/BASE_CLINICA_PREDICCION_FILTRADA'
# path_model_out = Path.home() + '/Tremor_machine_learning/Proyecto/LSTM_Prediction_Alex/'

print(path_data_base_in)
print(path_model_out)
# print(path_script + data_base)
# print(path_script + output_path)

#%% LOAD DATA
Train_set, Test_set, Val_set = loadFCN.load_database_2s(path_data_base_in, subset_size, normalize_method, random_seed)
Train_set.to(device)
Test_set.to(device)
Val_set.to(device)
#%% LSTM
class LSTM_Prediction(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers,device):
        super(LSTM_Prediction, self).__init__()
        self.input_size  =  input_size
        self.hidden_size =  hidden_size
        self.n_layers  =  n_layers
        self.output_size =  output_size
        self.device = device
        # self.bidirectional = bidirectional
        self.sigmoid = nn.Sigmoid()
        # Step1: the LSTM model
        self.lstm1 = nn.LSTM(input_size, hidden_size, n_layers-1, batch_first=True)#, bidirectional=bidirectional)
        # self.lstm2 = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True)#, bidirectional=bidirectional)

        self.fc = nn.Linear(hidden_size, output_size)
            
    def forward(self, signals, prints=False):
        if prints: print('signals shape:', signals.shape)
        

        # Hidden state:
        hidden_state_1 = torch.zeros(self.n_layers-1, signals.size(0), self.hidden_size).to(self.device)
        # hidden_state_2 = torch.zeros(self.n_layers, signals.size(0), self.hidden_size).to(self.device)
        # Cell state:
        cell_state_1 = torch.zeros(self.n_layers-1, signals.size(0), self.hidden_size).to(self.device)
        # cell_state_2 = torch.zeros(self.n_layers, signals.size(0), self.hidden_size).to(self.device)
        
        
        # LSTM:
        output, (last_hidden_state_1, last_cell_state_1) = self.lstm1(signals, (hidden_state_1, cell_state_1))
        # a = output.detach().numpy()
        # output = output[:,:,-1].reshape(output.shape[0],output.shape[1],-1)
        # a = output.detach().numpy()
        # output, (last_hidden_state_2, last_cell_state_2) = self.lstm2(output, (hidden_state_2, cell_state_2))
        # a = output.detach().numpy()
        # Reshape
        output = output[:, -1, :]
        # a = output.detach().numpy()
        # FNN:
        output = self.fc(output)
        output = self.sigmoid(output)
        # a = output.detach().numpy()
        return output


#%% TRAIN ALGORITHM

def train_network_with_val(model, train_data, val_data, batchSize=10, num_epochs=10,
                  train_samples = 50, hor_pred = 20,
                  l_r=0.001, file_path = '', best_valid_loss = float("Inf"),
                  save_name = 'model_name',gpu_cpu = device):
    
    '''Trains the model and computes the average accuracy for train and test data.
    If enabled, it also shows the loss and accuracy over the iterations.'''
    
    print('Get data ready...')
    # Create dataloader for training dataset - so we can train on multiple batches
    # Shuffle after every epoch
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batchSize, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset = val_data, batch_size = batchSize, shuffle = True)
    # Create criterion and optimizer 
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=l_r)
 
    # Losses & Iterations: to keep all losses during training (for plotting)
    losses = []
    iterations = []
    # Train and test accuracies: to keep their values also (for plotting)
    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []
    eval_every = len(train_loader)# // 2
    
    stop_counter = 0
    
    print('Training started...')
    iteration = 0
    # Train the data multiple times
    for epoch in range(num_epochs):
        # running_loss = 0
        for signals in iter(train_loader):
            # Set model in training mode:
            model.train()
            
            #Load input signal and reshape [n_batches, sequence_length, features]
            signals = signals.reshape(10, 100,-1)
            
            # Select train and label samples accordingly to Prediction Horizon
            signals_train = signals[:,0:train_samples,:].float().to(gpu_cpu)
            labels = torch.squeeze(signals[:,train_samples:train_samples+hor_pred]).float().to(gpu_cpu)#.type(torch.LongTensor)
            
            ## Convert to ndarray for debugging
            # aaa = signals_train.numpy()
            # aa = labels.numpy()          
            # a = signals_train.numpy()
            
            # Clears the gradients from previous iteration
            optimizer.zero_grad()
            # Create log probabilities
            out = model(signals_train)
            # a = out.detach().numpy()
            
            
            # Computes loss: how far is the prediction from the actual?
            loss = criterion(out, labels)
            # Computes gradients for neurons
            loss.backward()
            # Updates the weights
            optimizer.step()
            
            # Save information after this iteration
            running_loss += loss.item()
            global_step += 1
        # print(f'Epoch: {epoch:3d}. Loss: {running_loss/len(train_loader):.5f}')
            if global_step % eval_every == 0:
                model.eval()
                with torch.no_grad():
                    # validation loop
                    for signal_all_val in iter(val_loader):
                        #Load input signal and reshape [n_batches, sequence_length, features]
                        signal_all_val = signal_all_val.reshape(10, 100,-1)
                        
                        # Select train and label samples accordingly to Prediction Horizon
                        signals_val = signal_all_val[:,0:train_samples,:].float().to(gpu_cpu)
                        labels_val = torch.squeeze(signal_all_val[:,train_samples:train_samples+hor_pred]).float().to(gpu_cpu)#.type(torch.LongTensor)
                        
                        # Compute the output of the model and calculate loss
                        out = model(signals_val)
                        loss = criterion(out,labels_val)
                        
                        # Save loss
                        valid_running_loss += loss.item()
                        
                # Average train and valid loss
                average_train_loss = running_loss / eval_every
                average_valid_loss = valid_running_loss / len(val_loader)
                # Update loss values and steps
                train_loss_list.append(average_train_loss)
                valid_loss_list.append(average_valid_loss)
                global_steps_list.append(global_step)
            
                # Resetting running values
                running_loss = 0.0
                valid_running_loss = 0.0
                model.train()
                
                # print progress
                print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'
                      .format(epoch+1, num_epochs, global_step, num_epochs*len(train_loader),
                              average_train_loss, average_valid_loss))
    
                # checkpoint
                if best_valid_loss > average_valid_loss:
                    best_valid_loss = average_valid_loss
                    valFcn.save_checkpoint(file_path + save_name, model, optimizer, best_valid_loss)
                    valFcn.save_metrics(file_path[:-3] + save_name + '_metrics', train_loss_list, valid_loss_list, global_steps_list)
                    stop_counter = 0
                else:
                    stop_counter += 1
        #Try to stop the model if loss does not decrease during 5 validation steps
        if stop_counter > stop_treining_steps:
            valFcn.save_checkpoint(file_path + save_name, model, optimizer, best_valid_loss)
            valFcn.save_metrics(file_path[:-3] + save_name + '_metrics', train_loss_list, valid_loss_list, global_steps_list)
            break
            
    valFcn.save_checkpoint(file_path + save_name, model, optimizer, best_valid_loss)
    valFcn.save_metrics(file_path + save_name + '_metrics', train_loss_list, valid_loss_list, global_steps_list)
    print('Finished Training!')
    
#%% CREATE MODEL
train_loader_example = torch.utils.data.DataLoader(Train_set, batch_size=10)

# Creating the Model

#%% LOOP
start = time.time()
for i in range(model_ini,n_models):
    lstm_model= LSTM_Prediction(input_size, int(output_size[i]), int(hidden_size[i]), int(layer_size[i]),
                                device).float().to(device)

    print('lstm_example:', lstm_model, '\n')

#%% TRAIN NETWORK
    
    train_network_with_val(lstm_model, train_data = Train_set, val_data = Val_set, batchSize=batch_size,
              num_epochs=n_epochs, train_samples = int(train_window[i]), hor_pred = int(prediction_horizon[i]),
              l_r=float(learning_rate[i]), file_path = path_model_out, save_name = str(model_name[i]))
    
    print('MODEL NUMBER:', i, '\n')
    print('MODEL TRAINED:', str(model_name[i]), '\n')

end = time.time()
print(end - start)

#%% TEST FINAL MODEL

#Load best model
    # best_model = LSTM_Prediction(input_size, output_size[i], hidden_size[i], layer_size[i],device).float().to(device)
    # optimizer = torch.optim.Adam(best_model.parameters(), lr=learning_rate[i])

    # valFcn.load_checkpoint(path_model_out + model_name[i], best_model, optimizer, device)

    # correlation, mse, rmse, phase_delay_oof, phase_delay_lstm = valFcn.evaluate_pred_oof(best_model,
    #               Test_set, version='title', threshold=0.5,
    #               train_samples = train_window[i], hor_pred = prediction_horizon[i],
    #               path = path_model_out, file_name = model_name[i])


# gitFCN.git_push(r'/home/victorl/Tremor_machine_learning/.git','test_models')
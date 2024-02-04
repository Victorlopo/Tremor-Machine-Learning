# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 10:16:44 2021

@author: Usuario
"""

import numpy as np
import pandas as pd 
from os import listdir
from os.path import isfile, join
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
import math
from sklearn.preprocessing import normalize
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import random
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.autograd as autograd
#%%
mypath = "D:\\TREMOR_DATABASE\\Labelled\\"
# Extraigo el número de bloques totales dentro de la base de datos
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath,f))]
onlyfiles = sorted(onlyfiles)

# Número de segmentos totales de la base de datos
num_bloques = 20299
subset_size = 100;

# Creo un array con tantas filas como bloques hay, y 52 columnas (50 para ángulos, la 51 para el identificador y la 52 con la etiqueta (1 temblor, 0 no temblor))
conjunto_total = np.zeros((num_bloques, 52))
fila_conjunto_total = 0
# Recorro el onlyfiles para almacenar los angulos, id, y etiqueta en el conjunto_total
for i in onlyfiles:
    df = pd.read_csv(mypath + '\\' + i, sep=',')
    num_filas = df.shape[0]
    pos_bloque = 0
    for x in range(0,num_filas,1):
        id_segmento = str(i[6:12])
        identificador = id_segmento + '.' + str(pos_bloque)
        identificador = float(identificador)
        conjunto_total[fila_conjunto_total,0:-2] = df.iloc[x,1:51]
        conjunto_total[fila_conjunto_total,-1] = df.iloc[x,-1]
        conjunto_total[fila_conjunto_total,-2] = identificador
        pos_bloque+=1
        fila_conjunto_total += 1
        #print(x)

# Bucle for para situar todos los ángulos en 0 (les resto la media de sus valores)
for i in range(0,conjunto_total.shape[0],1):
    media = sum(conjunto_total[i,0:50])/50
    conjunto_total[i,0:50] = conjunto_total[i,0:50] - media

tremor_0 = np.squeeze(conjunto_total[np.where(conjunto_total[:,-1]==0),:])
tremor_1 = np.squeeze(conjunto_total[np.where(conjunto_total[:,-1]==1),:])    
tremor_0_subset = tremor_0[random.sample(range(0,len(tremor_0)), subset_size),:]
tremor_1_subset = tremor_0[random.sample(range(0,len(tremor_0)), subset_size),:]    
tremor_subset = np.concatenate((tremor_0_subset,tremor_1_subset))

# cjto_total_norm = normalize(conjunto_total[:,0:50])
# Creo el train_set y el test_set, el parámetro test_size es por lo que se multiplica el conjunto_total
# para formar el test_set. El random_state es la semilla
train_set, test_set = train_test_split(tremor_subset, test_size = 0.2, random_state = 220)
# Aquí divido los conjunto de entrenamiento y prueba entre 2 (o el número que fuera)
# porque mi ordenador no es capaz de procesar todo
# =============================================================================
# train_set = train_set[0:math.floor(int(train_set.shape[0])/int(10)),:]
# test_set = test_set[0:math.floor(int(test_set.shape[0])/int(10)),:]
# =============================================================================

# Separo valores y etiquetas de los conjuntos de entrenamiento y prueba por si lo necesito
train_set_norm = normalize(train_set[:,0:50])
test_set_norm = normalize(test_set[:,0:50])
X_train = torch.tensor(torch.from_numpy(train_set_norm)).type(torch.LongTensor)
Y_train = torch.from_numpy(train_set[:,-1]).type(torch.LongTensor)
X_test = torch.from_numpy(test_set_norm).type(torch.LongTensor)
Y_test = torch.from_numpy(test_set[:,-1])  .type(torch.LongTensor)


X_train_tensors_final = torch.reshape(X_train,   (X_train.shape[1], 1, X_train.shape[0]))
Y_train_tensors_final = Y_train
#%%
class LSTM_Classifier(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(LSTM_Classifier, self).__init__()
        self.output_size = output_size
        self.input_size = input_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_size, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)
        
    def forward(self, x, hidden):
        batch_size = x.size(0)
        x = x.long()
        lstm_out, hidden = self.lstm(x, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(lstm_out[:, -1, :])
        out = self.sigmoid(out)
        
        return out, hidden
    
    def init_hidden(self, batch_size):
        
        hidden= (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)).long(),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)).long())
        return hidden
    
#%%
# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
#%%
input_size = 1
output_size = 1
embedding_dim = 400
hidden_dim = 51
n_layers = 2

model = LSTM_Classifier(input_size, output_size, hidden_dim, n_layers)
model.to(device)

lr=0.005
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#%%
batch_size = 10
epochs = 5
counter = 0
print_every = 1000
clip = 5
valid_loss_min = np.Inf

model.train()
for i in range(epochs):
    for i in range(subset_size):
        
        X_train_batch = X_train_tensors_final[:,:,i]
        Y_train_batch = Y_train_tensors_final[i]
        X_train_batch = torch.reshape(X_train_batch,   (1,X_train_batch.shape[0],1))
        X_train_batch, Y_train_batch = X_train_batch.to(device), Y_train_batch.to(device)
        
        h = model.init_hidden(batch_size)
        
        model.zero_grad()
        output, h = model(X_train_batch, h)
        loss = criterion(output.squeeze(), Y_train_batch.float())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

            
        model.train()
    print("Epoch: {}/{}...".format(i+1, epochs), "Loss: {:.6f}...".format(loss.item()))

#%%

#%%

#%%

#%%

#%%


# =============================================================================

# =============================================================================



#%%
# class SentimentNet(nn.Module):
#     def __init__(self, input_size, output_size, hidden_dim, n_layers):
#         super(SentimentNet, self).__init__()
#         self.output_size = output_size
#         self.input_size = input_size
#         self.n_layers = n_layers
#         self.hidden_dim = hidden_dim
#         self.lstm = nn.LSTM(input_size, hidden_dim, n_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_dim, output_size)
        
#     def forward(self, x, hidden):
#         batch_size = x.size(0)
#         x = x.long()
#         lstm_out, hidden = self.lstm(x, hidden)
#         lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
#         out = self.fc(lstm_out[:, -1, :])
#         out = self.sigmoid(out)
        
#         return out, hidden
    
#     def init_hidden(self, batch_size):
        
#         hidden= (autograd.Variable(torch.zeros(2, 1, self.hidden_dim)).long(),
#                 autograd.Variable(torch.zeros(2, 1, self.hidden_dim)).long())
#         return hidden
    
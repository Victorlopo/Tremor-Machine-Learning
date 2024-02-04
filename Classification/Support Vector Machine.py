# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 11:12:52 2021

@author: Usuario
"""

import numpy as np
import pandas as pd 
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score


mypath = "C:\\Users\\Usuario\\Desktop\\UNIVERSIDAD\\TFG\\BASE_DE_DATOS_FILTRADA\\"
# Extraigo el número de bloques totales dentro de la base de datos
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath,f))]
onlyfiles = sorted(onlyfiles)
num_bloques = 0
for i in onlyfiles:
    df = pd.read_csv(mypath + '\\' + i, sep=',')
    num_filas = df.shape[0]
    for x in range(0,num_filas,1):
        num_bloques += 1
        print(num_bloques)

conjunto_total = np.zeros((num_bloques, 4))
fila_conjunto_total = 0
for i in onlyfiles:
    df = pd.read_csv(mypath + '\\' + i, sep=',')
    num_filas = df.shape[0]
    pos_bloque = 0
    for x in range(0,num_filas,1):
        id_segmento = str(i[6:12])
        identificador = id_segmento + '.' + str(pos_bloque)
        identificador = float(identificador)
        conjunto_total[fila_conjunto_total,3] = identificador
        conjunto_total[fila_conjunto_total,0] = df.iloc[x,51]
        conjunto_total[fila_conjunto_total,1] = df.iloc[x,53]
        conjunto_total[fila_conjunto_total,2] = df.iloc[x,-1]
        pos_bloque+=1
        fila_conjunto_total += 1

# =============================================================================
# conjunto_total = conjunto_total.tolist()
# =============================================================================
# =============================================================================
# id_segmento.toInt()
# =============================================================================
train_set, test_set = train_test_split(conjunto_total, test_size = 0.2, random_state = 220)
# =============================================================================
# vector_base = np.arange(0,num_bloques,1).tolist()
# valores_random = random.sample(vector_base, 3000)
# fila_train = 0
# num_no_temblor = 0
# num_temblor = 0
# for i in valores_random:
#     train[fila_train,:] = conjunto_total[i,:]   
#     if (train[fila_train,-1] == 0):
#         num_no_temblor += 1
#     else:
#         num_temblor += 1
#     fila_train += 1
# =============================================================================
vector_seleccion = np.array([0,1,3])
Y_train_id = train_set[:,2:]
Y_test_id = test_set[:,2:]
X_train = train_set[:,0:2]
Y_train = train_set[:,-2]
X_test = test_set[:,0:2]
Y_test = test_set[:,-2]

svm_clf = Pipeline([
        ('scaler', StandardScaler()),
        ('linear_svc', LinearSVC(C=50, loss='hinge')),
        ])

svm_clf.fit(X_train,Y_train)

Y_test_pred = cross_val_predict(svm_clf,X_test, Y_test, cv=3)
Y_test_pred_id = np.zeros((len(Y_test_pred), 2))
Y_test_pred_id[:,0] = Y_test_pred[:]
Y_test_pred_id[:,1] = Y_test_id[:,-1]
True_False = (Y_test_pred == Y_test)
posiciones_false = np.where(True_False == False)
matriz_confusion = confusion_matrix(Y_test , Y_test_pred)

precision = precision_score(Y_test, Y_test_pred)
sensibilidad = recall_score(Y_test, Y_test_pred)

# =============================================================================
# plt.figure()
# # Etiquetas para los ejes
# plt.scatter(X_train[:,1], Y_train, marker = 'o')
# plt.xlabel('fft')
# plt.ylabel('Temblor/No temblor')
#         
# plt.figure()
# # plotea la decision funcion
# ax = plt.gca()
# xlim = ax.get_xlim()
# ylim = ax.get_ylim()
#  
# # crea la malla para evaluar el modelo
# xx = np.linspace(xlim[0], xlim[1], 30)
# yy = np.linspace(ylim[0], ylim[1], 30)
# YY, XX = np.meshgrid(yy, xx)
# xy = np.vstack([XX.ravel(), YY.ravel()]).T
# Z = svm_clf.decision_function(xy).reshape(XX.shape)
# 
# # plotea decision margenes y fronteras
# ax.contour(XX, YY, Z, colors='g', levels=[-1, 0, 1], alpha=0.5, linestyles=['-.', '-', '-.'])
# # =============================================================================
# # # plotea support vectors
# # ax.scatter(svm_clf.support_vectors_[:, 0], svm_clf.support_vectors_[:, 1], s=100, linewidth=1, facecolors='none')
# # =============================================================================
# plt.show()
#       
# plt.figure()
# plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train, s=30, cmap=plt.cm.Paired)  
# # plotea la decision funcion
# ax = plt.gca()
# xlim = ax.get_xlim()
# ylim = ax.get_ylim()
# 
# 
# # crea la malla para evaluar el modelo
# xx = np.linspace(xlim[0], xlim[1], 30)
# yy = np.linspace(ylim[0], ylim[1], 30)
# YY, XX = np.meshgrid(yy, xx)
# xy = np.vstack([XX.ravel(), YY.ravel()]).T
# Z = svm_clf.decision_function(xy).reshape(XX.shape)
# # plotea decision margenes y fronteras
# ax.contour(XX, YY, Z, colors='g', levels=[-1, 0, 1], alpha=0.5, linestyles=['-.', '-', '-.'])
# # plotea support vectors
# # =============================================================================
# # ax.scatter(svm_clf.support_vectors_[:, 0], svm_clf.support_vectors_[:, 1], s=100, linewidth=1, facecolors='none')
# # plt.show()
# # =============================================================================
# =============================================================================
        

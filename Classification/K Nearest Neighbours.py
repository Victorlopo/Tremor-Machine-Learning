# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 12:53:10 2021

@author: Victor Lopo Martinez

"""

import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.preprocessing import normalize
from tslearn.neighbors import KNeighborsTimeSeriesClassifier

mypath = "my_path"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
onlyfiles = sorted(onlyfiles)

num_bloques = 20299
conjunto_total = np.zeros((num_bloques, 52))
fila_conjunto_total = 0
for i in onlyfiles:
    df = pd.read_csv(mypath + '\\' + i, sep=',')
    num_filas = df.shape[0]
    for x in range(num_filas):
        id_segmento = str(i[6:12])
        identificador = id_segmento + '.' + str(x)  # Suponiendo que pos_bloque es igual a x en este contexto
        identificador = float(identificador)
        conjunto_total[fila_conjunto_total,0:-2] = df.iloc[x,1:51]
        conjunto_total[fila_conjunto_total,-2] = identificador
        conjunto_total[fila_conjunto_total,-1] = df.iloc[x,-1]
        fila_conjunto_total += 1

# Dividir el conjunto de datos antes de normalizar
train_set, test_set = train_test_split(conjunto_total, test_size=0.2, random_state=220)

# Normalizar los conjuntos de entrenamiento y prueba
X_train_norm = normalize(train_set[:,0:50])  # Normaliza el conjunto de entrenamiento
Y_train = train_set[:,-1]
X_test_norm = normalize(test_set[:,0:50])  # Normaliza el conjunto de prueba con el mismo criterio
Y_test = test_set[:,-1]

# Nearest neighbor classification
knn_clf = KNeighborsTimeSeriesClassifier(n_neighbors=3, metric="dtw")
knn_clf.fit(X_train_norm, Y_train)
predicted_labels = knn_clf.predict(X_test_norm)

# Cálculo de la matriz de confusión, precisión y sensibilidad
matriz_confusion_knn = confusion_matrix(Y_test, predicted_labels)
precision_knn = precision_score(Y_test, predicted_labels)
sensibilidad_knn = recall_score(Y_test, predicted_labels)

print("Matriz de confusión:\n", matriz_confusion_knn)
print("Precisión:", precision_knn)
print("Sensibilidad:", sensibilidad_knn)

# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 09:21:39 2021

@author: Víctor Lopo Martínez

Clasificador RandomForest con 500 estimadores y máximo número de nodos = 16
Base de datos etiquetada clínicamente
Entrada al clasificador -> 50 angulos filtrados y situados en 0
Precisión = 95
Sensibilidad = 92

"""

import numpy as np
import pandas as pd 
from os import listdir
from os.path import isfile, join
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
import math
from sklearn.ensemble import RandomForestClassifier


mypath = "C:\\Users\\Usuario\\Desktop\\UNIVERSIDAD\\TFG\\BASE_DE_DATOS_FILTRADA_NUEVA_PSD\\"
# Extraigo el número de bloques totales dentro de la base de datos
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath,f))]
onlyfiles = sorted(onlyfiles)

# Número de segmentos totales de la base de datos
num_bloques = 20299
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
        print(x)

# Bucle for para situar todos los ángulos en 0 (les resto la media de sus valores)
for i in range(0,conjunto_total.shape[0],1):
    media = sum(conjunto_total[i,0:50])/50
    conjunto_total[i,0:50] = conjunto_total[i,0:50] - media

# Creo el train_set y el test_set, el parámetro test_size es por lo que se multiplica el conjunto_total
# para formar el test_set. El random_state es la semilla
train_set, test_set = train_test_split(conjunto_total, test_size = 0.2, random_state = 220)
# Aquí divido los conjunto de entrenamiento y prueba entre 2 (o el número que fuera)
# porque mi ordenador no es capaz de procesar todo
# =============================================================================
# train_set = train_set[0:math.floor(int(train_set.shape[0])/int(2)),:]
# test_set = test_set[0:math.floor(int(test_set.shape[0])/int(2)),:]
# =============================================================================

# Separo valores y etiquetas de los conjuntos de entrenamiento y prueba por si lo necesito
X_train = train_set[:,0:50] 
Y_train = train_set[:,-1]
X_test = test_set[:,0:50]
Y_test = test_set[:,-1]   


# Creo el modelo del random forest
rnd_clf = RandomForestClassifier(n_estimators = 500, max_leaf_nodes =16, n_jobs=-1 )
rnd_clf.fit(X_train, Y_train)
y_pred_rf = rnd_clf.predict(X_test)

# Calculo la matriz de confusión
matriz_confusion_rf = confusion_matrix(Y_test, y_pred_rf)


# Calculo la precisión y sensibilidad en base a los valores obtenidos
precision_rf = precision_score(Y_test, y_pred_rf)
sensibilidad_rf = recall_score(Y_test, y_pred_rf)




























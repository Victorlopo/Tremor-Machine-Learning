# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 09:21:39 2021

@author: Víctor Lopo Martínez

"""

# Importaciones necesarias
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import normalize

# Ruta al directorio con los datos
mypath = "my_path"

# Extracción de archivos de la base de datos
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
onlyfiles = sorted(onlyfiles)

# Inicialización del conjunto de datos
num_bloques = 20299  # Número total de segmentos en la base de datos
conjunto_total = np.zeros((num_bloques, 52))  # Array para almacenar ángulos, identificador y etiqueta

# Lectura y almacenamiento de datos
fila_conjunto_total = 0
for i in onlyfiles:
    df = pd.read_csv(join(mypath, i), sep=',')
    for x, row in df.iterrows():
        identificador = float(f"{i[6:12]}.{x}")
        conjunto_total[fila_conjunto_total, :-2] = row[1:51]
        conjunto_total[fila_conjunto_total, -2] = identificador
        conjunto_total[fila_conjunto_total, -1] = row[-1]
        fila_conjunto_total += 1

# División del conjunto en entrenamiento y prueba
train_set, test_set = train_test_split(conjunto_total, test_size=0.2, random_state=220)

# Normalización de los conjuntos de entrenamiento y prueba (después del split)
X_train = normalize(train_set[:, 0:50])
Y_train = train_set[:, -1]
X_test = normalize(test_set[:, 0:50])
Y_test = test_set[:, -1]

# Configuración de la validación cruzada para la selección de hiperparámetros óptimos
param_grid = {
    'n_estimators': [100, 300, 500],
    'max_leaf_nodes': [16, 32, 64],
    'max_depth': [None, 10, 20]
}
grid_search = GridSearchCV(RandomForestClassifier(n_jobs=-1), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, Y_train)

# Resultados de la validación cruzada
print("Mejores parámetros:", grid_search.best_params_)
best_model = grid_search.best_estimator_

# Predicciones y evaluación del modelo
y_pred_rf = best_model.predict(X_test)
matriz_confusion_rf = confusion_matrix(Y_test, y_pred_rf)
precision_rf = precision_score(Y_test, y_pred_rf)
sensibilidad_rf = recall_score(Y_test, y_pred_rf)

# Análisis de resultados
print("Matriz de confusión:\n", matriz_confusion_rf)
print("Precisión:", precision_rf)
print("Sensibilidad:", sensibilidad_rf)

# Al ser un análisis de una serie de temporal no tiene sentido hacer un análisis de importancia de las características porque todos 
# los valores del vector de características del input representan la misma variación angular a lo largo del tiempo.

# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 11:12:52 2021

@author: Víctor Lopo Martínez
"""

import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.preprocessing import normalize
from sklearn.svm import SVC

mypath = "my_path"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
onlyfiles = sorted(onlyfiles)

num_bloques = 20299
conjunto_total = np.zeros((num_bloques, 52))
fila_conjunto_total = 0
for i in onlyfiles:
    df = pd.read_csv(join(mypath, i), sep=',')
    for x in range(df.shape[0]):
        identificador = float(f"{i[6:12]}.{x}")
        conjunto_total[fila_conjunto_total, 0:-2] = df.iloc[x, 1:51]
        conjunto_total[fila_conjunto_total, -2] = identificador
        conjunto_total[fila_conjunto_total, -1] = df.iloc[x, -1]
        fila_conjunto_total += 1

# Dividir el conjunto de datos antes de normalizar
train_set, test_set = train_test_split(conjunto_total, test_size=0.2, random_state=220)

# Normalizar los conjuntos de entrenamiento y prueba
X_train_norm = normalize(train_set[:, 0:50])
Y_train = train_set[:, -1]
X_test_norm = normalize(test_set[:, 0:50])
Y_test = test_set[:, -1]

# Configuración de la validación cruzada para la selección de hiperparámetros óptimos
param_grid = {
    'C': [0.1, 1, 10, 100],  # Valores de C para regularización
    'gamma': ['scale', 'auto', 0.1, 1, 10, 100],  # Coeficiente para el kernel RBF
    'kernel': ['linear', 'rbf', 'poly']  # Tipos de kernel
}

svm_clf = SVC()
grid_search = GridSearchCV(svm_clf, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_norm, Y_train)

# Mejores parámetros y modelo
print("Mejores parámetros:", grid_search.best_params_)
best_model = grid_search.best_estimator_

# Predicciones con el modelo optimizado
predicted_labels = best_model.predict(X_test_norm)

# Cálculo de la matriz de confusión, precisión y sensibilidad
matriz_confusion = confusion_matrix(Y_test, predicted_labels)
precision = precision_score(Y_test, predicted_labels)
sensibilidad = recall_score(Y_test, predicted_labels)

print("Matriz de confusión:\n", matriz_confusion)
print("Precisión:", precision)
print("Sensibilidad:", sensibilidad)

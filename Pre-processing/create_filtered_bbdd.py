# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 14:15:19 2021

@author: Usuario
"""

import numpy as np
import pandas as pd
import filtros
from os import listdir
import math
from scipy import signal
from os.path import isfile, join
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq

mypath = "C:\\Users\\Usuario\\Desktop\\UNIVERSIDAD\\TFG\\DATA_ALL_TRIALS_NUEVA"
outpath = "C:\\Users\\Usuario\\Desktop\\UNIVERSIDAD\\TFG\\BASE_FINAL_FILTRADA\\"

tiempo_bloque = 1
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath,f))]
onlyfiles = sorted(onlyfiles)
derecha = 'Right_'
izquierda = 'lefft_'
# Umbral psd y fft para una variación pico pico de 0.4 grados de temblor
umbral_psd = 0.006
umbral_fft =  0.1502

# Vamos leyendo la carpeta de mypath para ir seleccionando automáticamente los ficheros
for i in onlyfiles:
    # Hago un datagrama con el fichero que se encuentre dentro del bucle for
    df = pd.read_csv(mypath + '\\' + i, sep=',')
    # Extraigo parámetros del dataframe para facilitar su uso después
    muestras_bloque = 1*int(df['fs_IMU'][0])
    num_muestras_on = int(df['STIM_ON_POS_IMU'][1]) - int(df['STIM_ON_POS_IMU'][0])
    num_muestras_off = int(df['STIM_OFF_POS_IMU'][1]) - int(df['STIM_OFF_POS_IMU'][0])
    num_bloques_on = int(math.floor(num_muestras_on/muestras_bloque))
    num_bloques_off = int(math.floor(num_muestras_off/muestras_bloque))
    num_bloques_totales = num_bloques_off + num_bloques_on
    STIM_ON_POS_IMU_1 = int(df['STIM_ON_POS_IMU'][1])
    STIM_ON_POS_IMU_0 = int(df['STIM_ON_POS_IMU'][0])
    STIM_OFF_POS_IMU_1 = int(df['STIM_OFF_POS_IMU'][1])
    STIM_OFF_POS_IMU_0 = int(df['STIM_OFF_POS_IMU'][0])
    id_archivo = int(i[0:6])

    # Creo un dataframe con valores a 0 para rellenarlo y pasarlo a csv después
    df1 = pd.DataFrame(np.zeros((num_bloques_totales, 62)))
    cuenta = 0
    derecha = 'Right_'
    izquierda = 'Lefft_'
    # Compruebo que hay valores en los ángulos de la muñeca izquierda
    if (int(df['ANGLE_LEFT_WRIST_FE'][20]) != 0):
        # Compruebo que se han cogido datos en el periodo con estimulación
        if (STIM_ON_POS_IMU_1 - STIM_ON_POS_IMU_0 > muestras_bloque):
            # Añado las etiquetas identificativas del bloque a partir del df
            df1.iloc[cuenta,-7] = str(df['SIDE_TREMOR'][0])
            df1.iloc[cuenta,-6] = str(df['TRIAL_ELECTRODE'][0])
            df1.iloc[cuenta,-5] = str(df['TRIAL_POSTURE'][0])
            df1.iloc[cuenta,-4] = str(df['SUBJECT'][0])
            df1.iloc[cuenta,-8] = int(df['TRIAL_DATE'][0])
            # Hago los bloques de 1 seg de duración, la densidad espectral de potencia y la fft
            for x in range(STIM_ON_POS_IMU_0,STIM_ON_POS_IMU_1+1, muestras_bloque):
                if (STIM_ON_POS_IMU_1 - x >= muestras_bloque):
                    # Añado los bloques de ángulos
                    df1.iloc[cuenta, 0: muestras_bloque] = df['ANGLE_LEFT_WRIST_FE'][int(x): int(x) +muestras_bloque].values.tolist()
                    # Paso a numpy el conjunto de ángulos para hacer la fft y la psd
                    a = df1.iloc[cuenta,0:muestras_bloque].to_numpy()
                    a = a.astype(np.float)
                    a = filtros.butter_bandpass_filter(a,4,10,50,3)
                    df1.iloc[cuenta,0:muestras_bloque] = a
                    # Obtengo las freq y la psd del bloque de ángulos (ventana de 50 muestras = total)
                    freq_psd, psd = signal.welch(a,50,nperseg=muestras_bloque)
                    # Obtengo el máximo de la psd entre 3 y 9 Hz y lo añado al df1 junto con la freq a la que aparece
                    max_psd = max(psd[4:10])
                    df1.iloc[cuenta,50] = max_psd
                    df1.iloc[cuenta,51] = psd.tolist().index(max_psd) 
                    # Comparo el valor de la psd con el umbral para asignar temblor (1) o no temblor (0)
                    if (max_psd < umbral_psd):
                        df1.iloc[cuenta,-3] = 0
                    else:
                        df1.iloc[cuenta,-3] = 1
                    # Obtengo la fft y la freq a la que aparece y lo añado al df1
                    fast_fourier = 2.0/50 * np.abs(fft(a)[0:50//2]) 
                    freq_fft = fftfreq(50,1/50)[:50//2]
                    max_fft = max(fast_fourier[4:10])
                    df1.iloc[cuenta,52] = max_fft
                    df1.iloc[cuenta,53] = fast_fourier.tolist().index(max_fft)
                    # Comparo con el umbral de fft (0.4 grados pico pico) y asigno temblor (1) o no temblor (0)
                    if (max_fft < umbral_fft):
                        df1.iloc[cuenta,-2] = 0
                    else:
                        df1.iloc[cuenta,-2] = 1
                    # Etiqueto clínicamente
                    if (id_archivo < 1000):
                        df1.iloc[cuenta,-1] = 1
                    else:
                        df1.iloc[cuenta,-1] = 0
                    
                    cuenta += 1
        # Compruebo que se han recogido datos en el periodo sin estimulación
        if (STIM_OFF_POS_IMU_1 - STIM_OFF_POS_IMU_0 > muestras_bloque):
            # Añado etiquetas identificadoras del bloque
            df1.iloc[cuenta,-7] = str(df['SIDE_TREMOR'][0])
            df1.iloc[cuenta,-6] = str(df['TRIAL_ELECTRODE'][0])
            df1.iloc[cuenta,-5] = str(df['TRIAL_POSTURE'][0])
            df1.iloc[cuenta,-4] = str(df['SUBJECT'][0])
            df1.iloc[cuenta,-8] = int(df['TRIAL_DATE'][0])
            for x in range(STIM_OFF_POS_IMU_0,STIM_OFF_POS_IMU_1+1, muestras_bloque):
                if (STIM_OFF_POS_IMU_1 - x >= muestras_bloque):
                    # Coloco los ángulos obtenidos en el df
                    df1.iloc[cuenta, 0: muestras_bloque]= df['ANGLE_LEFT_WRIST_FE'][int(x): int(x) +muestras_bloque].values.tolist()
                    # Paso ese determinado bloque a numpy para trabajar más fácilmente con él
                    a = df1.iloc[cuenta, 0: muestras_bloque].to_numpy()
                    a = a.astype(np.float)
                    a = filtros.butter_bandpass_filter(a,4,10,50,3)
                    df1.iloc[cuenta,0:muestras_bloque] = a
                    # Obtengo la psd y las freq del bloque
                    freq_psd, psd = signal.welch(a,50,nperseg=50)
                    # Obtengo el máximo y lo añado a df1 junto con la freq a la que aparece
                    max_psd = max(psd[3:10])
                    df1.iloc[cuenta,50] = max_psd
                    df1.iloc[cuenta,51] = psd.tolist().index(max_psd)
                    # Añado la etiqueta de temblor (1) o no temblor(0) en función de la psd
                    if (max_psd < umbral_psd):
                        df1.iloc[cuenta,-3] = 0
                    else:
                        df1.iloc[cuenta,-3] = 1
                    # Obtengo la fft y las freq del bloque de ángulos
                    fast_fourier = 2.0/50 * np.abs(fft(a)[0:50//2]) 
                    freq_fft = fftfreq(50,1/50)[:50//2]
                    # Obtengo el máximo y la freq a la que aparece
                    max_fft = max(fast_fourier[3:10])
                    df1.iloc[cuenta,52] = max_fft
                    df1.iloc[cuenta,53] = fast_fourier.tolist().index(max_fft)
                    # Comparo el máximo con el umbral de la fft y asigno temblor(1) o no temblor (0)
                    if (max_fft < umbral_fft):
                        df1.iloc[cuenta,-2] = 0
                    else:
                        df1.iloc[cuenta,-2] = 1
                    # Etiqueto clínicamente
                    if (id_archivo < 1000):
                        df1.iloc[cuenta,-1] = 1
                    else:
                        df1.iloc[cuenta,-1] = 0
                    cuenta += 1
        # Creo el fichero csv de este datagrama            
        df1.to_csv(outpath + izquierda + i , sep= ',')
        
    # Mismo procedimiento que con la muñeca izquierda pero en la derecha
    if (int(df['ANGLE_RIGHT_WRIST_FE'][20]) != 0):
        cuenta = 0   
        # Compruebo que hay valores guardados durante la estimulación
        if (STIM_ON_POS_IMU_1 - STIM_ON_POS_IMU_0 > muestras_bloque):

            df1.iloc[cuenta,-7] = str(df['SIDE_TREMOR'][0])
            df1.iloc[cuenta,-6] = str(df['TRIAL_ELECTRODE'][0])
            df1.iloc[cuenta,-5] = str(df['TRIAL_POSTURE'][0])
            df1.iloc[cuenta,-4] = str(df['SUBJECT'][0])
            df1.iloc[cuenta,-8] = int(df['TRIAL_DATE'][0])
            # Hago los bloques de 1 seg de duración, la densidad espectral de potencia y la fft
            for x in range(STIM_ON_POS_IMU_0,STIM_ON_POS_IMU_1+1, muestras_bloque):
                if (STIM_ON_POS_IMU_1 - x >= muestras_bloque):
                    df1.iloc[cuenta, 0: muestras_bloque] = df['ANGLE_RIGHT_WRIST_FE'][int(x): int(x) +muestras_bloque].values.tolist()
                    a = df1.iloc[cuenta,0:muestras_bloque].to_numpy()
                    a = a.astype(np.float)
                    a = filtros.butter_bandpass_filter(a,4,10,50,3)
                    df1.iloc[cuenta,0:muestras_bloque] = a
                    freq_psd, psd = signal.welch(a,50,nperseg=muestras_bloque)
                    max_psd = max(psd[3:10])
                    df1.iloc[cuenta,50] = max_psd
                    df1.iloc[cuenta,51] = psd.tolist().index(max_psd) 
                    if (max_psd < umbral_psd):
                        df1.iloc[cuenta,-3] = 0
                    else:
                        df1.iloc[cuenta,-3] = 1
                    fast_fourier = 2.0/50 * np.abs(fft(a)[0:50//2]) 
                    freq_fft = fftfreq(50,1/50)[:50//2]
                    max_fft = max(fast_fourier[3:10])
                    df1.iloc[cuenta,52] = max_fft
                    df1.iloc[cuenta,53] = fast_fourier.tolist().index(max_fft)
                    if (max_fft < umbral_fft):
                        df1.iloc[cuenta,-2] = 0
                    else:
                        df1.iloc[cuenta,-2] = 1
                    # Etiquetado clínicamente
                    if (id_archivo < 1000):
                        df1.iloc[cuenta,-1] = 1
                    else:
                        df1.iloc[cuenta,-1] = 0
                    
                    cuenta += 1
        # Compruebo que hay valores guardados fuera de la estimulación
        if (STIM_OFF_POS_IMU_1 - STIM_OFF_POS_IMU_0 > muestras_bloque):
            df1.iloc[cuenta,-7] = str(df['SIDE_TREMOR'][0])
            df1.iloc[cuenta,-6] = str(df['TRIAL_ELECTRODE'][0])
            df1.iloc[cuenta,-5] = str(df['TRIAL_POSTURE'][0])
            df1.iloc[cuenta,-4] = str(df['SUBJECT'][0])
            df1.iloc[cuenta,-8] = int(df['TRIAL_DATE'][0])
            for x in range(STIM_OFF_POS_IMU_0,STIM_OFF_POS_IMU_1+1, muestras_bloque):
                if (STIM_OFF_POS_IMU_1 - x >= muestras_bloque):
                    df1.iloc[cuenta, 0: muestras_bloque]= df['ANGLE_RIGHT_WRIST_FE'][int(x): int(x) +muestras_bloque].values.tolist()
                    a = df1.iloc[cuenta, 0: muestras_bloque].to_numpy()
                    a = a.astype(np.float)
                    a = filtros.butter_bandpass_filter(a,4,10,50,3)
                    df1.iloc[cuenta,0:muestras_bloque] = a
                    freq_psd, psd = signal.welch(a,50,nperseg=50)
                    max_psd = max(psd[3:10])
                    df1.iloc[cuenta,50] = max_psd
                    df1.iloc[cuenta,51] = psd.tolist().index(max_psd)
                    if (max_psd < umbral_psd):
                        df1.iloc[cuenta,-3] = 0
                    else:
                        df1.iloc[cuenta,-3] = 1
                    fast_fourier = 2.0/50 * np.abs(fft(a)[0:50//2]) 
                    freq_fft = fftfreq(50,1/50)[:50//2]
                    max_fft = max(fast_fourier[3:10])
                    df1.iloc[cuenta,52] = max_fft
                    df1.iloc[cuenta,53] = fast_fourier.tolist().index(max_fft)
                    if (max_fft < umbral_fft):
                        df1.iloc[cuenta,-2] = 0
                    else:
                        df1.iloc[cuenta,-2] = 1
                    # Etiquetado clínicamente
                    if (id_archivo < 1000):
                        df1.iloc[cuenta,-1] = 1
                    else:
                        df1.iloc[cuenta,-1] = 0
                    cuenta += 1
        # Convertir el dataframe a csv          
        df1.to_csv(outpath + derecha + i , sep= ',')
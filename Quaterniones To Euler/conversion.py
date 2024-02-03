# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 12:18:39 2020

@author: Usuario
"""
import scipy.io
import csv
import os
import Transforms as trans
import numpy as np
from scipy.spatial.transform import Rotation as R
from tkinter import *
from tkinter import filedialog
import matplotlib.pyplot as plt

fs_IMU = 50
ventana = Tk()
current_file_emg = filedialog.askopenfilename() 
ventana.withdraw()
separate = os.path.split(current_file_emg)
nombre_fichero = separate[1]
IMUS_importado = scipy.io.loadmat(nombre_fichero)
Datos = IMUS_importado.get('Quaternions_raw')
a = Datos[0,0]
idx = [1,2,3,0]
hand_quat = Datos[0,2][:,idx]
fore_quat = Datos[0,3][:,idx]
r_hand = R.from_quat(hand_quat)
r_fore = R.from_quat(fore_quat)
r_hand1 = r_hand.as_matrix()
r_fore1 = r_fore.as_matrix()

wrist_angles_r, rot_mat = trans.calc_rot_angles_euler(r_fore1, r_hand1, seq='YZX')
wrist_flexion = wrist_angles_r[0]*(-1)


time = np.linspace(0,len(wrist_angles_r)/50, len(wrist_angles_r))
wrist_angles_r[:,0] = -wrist_angles_r[:,0]
wrist_angles_r[:,[1,2]] = wrist_angles_r[:,[2,1]]
wrist_angles_r[:,1] = -wrist_angles_r[:,1]
wrist_angles_r[:,2] = -wrist_angles_r[:,2]
with open('tabla.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(wrist_angles_r)
plt.plot(time, wrist_angles_r[:,2])


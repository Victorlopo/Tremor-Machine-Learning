# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from scipy.signal import butter, lfilter
from scipy import signal

def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def high_pass_angle (highcut, fs_IMU, order):
    nyq=0.5 * fs_IMU
    high = highcut/nyq
    HP_IMU_b, HP_IMU_a= butter(order, high, btype='high')
    return HP_IMU_b, HP_IMU_a

def high_pass_filter(data, highcut,fs_IMU, order ):
    b, a = high_pass_angle(highcut, fs_IMU, order)
    y = signal.filtfilt(b,a,data)
    return y

def low_pass_angle (lowcut, fs_IMU, order ):
    nyq=0.5 * fs_IMU
    low = lowcut/nyq
    b, a= butter(order, low, btype='low')
    return b,a

def low_pass_filter (data, lowcut, fs_IMU, order ):
    b,a = low_pass_angle(lowcut, fs_IMU, order)
    y = signal.filtfilt(b,a,data)
    return y 
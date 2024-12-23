import os
import random
import numpy as np
from scipy.signal import stft
import tensorflow as tf

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def fft(data, dt):
    f = np.fft.fft(data)
    freq = np.fft.fftfreq(data.shape[0], dt)
    amp = np.abs(f/(data.shape[0]/2))
    plt_lim = int(data.shape[0]/2)
    return freq[1:plt_lim], amp[1:plt_lim]

def short_time_ft(data, dt, nperseg):
    return stft(x=data, fs=1/dt, nperseg=nperseg)
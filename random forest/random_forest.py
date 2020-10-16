#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import tensorflow as tf
from keras.layers import SimpleRNN, LSTM, GRU, Bidirectional, Conv1D, MaxPooling1D, Dropout
from keras.models import Sequential
from keras.layers import Dense
import os
import random
import include_rfr as rfr

# %%

'''Data pre-processing'''

data = pd.io.parsers.read_csv('../data/LD2011_2014.txt', sep=";", index_col=0, header=0, low_memory=False, decimal=',')
df = data
df = df.iloc[2*96:, :]
df = df.iloc[:-1, :]
df = df.iloc[:-3*96, :]

df.index = pd.to_datetime(df.index, format='%Y-%m-%d')
df = df.groupby(pd.Grouper(freq='W-MON')).sum()
df = df.iloc[:-1, :]

print(df.transpose())

# %%

periods_to_forecast = 4
print(periods_to_forecast)

X_original = df.transpose().values
n = np.shape(X_original)[0]
d = np.shape(X_original)[1]
d0 = d-periods_to_forecast
X_train = X_original[:, :d0]
X_test = X_original[:, d0:d0+periods_to_forecast] 

tau = np.argmax(X_original != 0, axis=1)

current_directory = os.getcwd()
trajectories_dir = os.path.join(current_directory, r'trajectories')
if not os.path.exists(trajectories_dir):
    os.makedirs(trajectories_dir)

# %%

X_pred = np.zeros((n, d-d0))
        
for i in range(n):
    np.random.seed(0)
    tf.set_random_seed(1)
    print('processing MT:', i+1)
    A = X_train[i, tau[i]:]
    X_pred[i, :] = rfr.ext_archetype(d, d0, A, time_window=np.minimum(16, int(np.floor(np.shape(A)[0]/2))), model_dl=LSTM, cnn=False, rfr=True)  
    print('RRMSE =', rfr.rrmse(X_test[i, :], X_pred[i, :]))
    print('forecasts:', X_pred[i, :])
    print('test:', X_test[i, :])
            
print('RRMSE =', rfr.rrmse(X_test, X_pred))
print('MPE =', rfr.mpe(X_test, X_pred))

save_file_forecast_best = ('matrix_forecast_best_plain.csv')
np.savetxt(save_file_forecast_best, X_pred, delimiter=',')

# %%

trajectory_list = random.sample(range(n), 5)

for i in trajectory_list:
    rfr.trajectory_plot(i, X_test, X_pred)
    print(rfr.mpe(X_test[i, :], X_pred[i, :]))

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.ndimage.interpolation import shift
from sklearn.metrics import mean_absolute_error
import include_amf as amf
import os
import random

np.random.seed(0)

# %%

'''Data pre-processing'''

data = pd.io.parsers.read_csv('LD2011_2014.txt', sep=";", index_col=0, header=0, low_memory=False, decimal=',')
df = data
df = df.iloc[2*96:, :]
df = df.iloc[:-1, :]
df = df.iloc[:-3*96, :]

df.index = pd.to_datetime(df.index, format='%Y-%m-%d')
df = df.groupby(pd.Grouper(freq='W-MON')).sum()
df = df.iloc[:-1, :]

print(df.transpose())

current_directory = os.getcwd()
mask_dir = os.path.join(current_directory, r'mask_test')
if not os.path.exists(mask_dir):
    os.makedirs(mask_dir)
trajectories_dir = os.path.join(current_directory, r'trajectories')
if not os.path.exists(trajectories_dir):
    os.makedirs(trajectories_dir)

# %%

periods_to_forecast = 4

X_original = df.transpose().values
n = np.shape(X_original)[0]
d = np.shape(X_original)[1]
d0 = d-periods_to_forecast
X_train = X_original[:, :d0]
X_test = X_original[:, d0:d0+periods_to_forecast] 

# %%

best_error = 1000000
f = open("mask_test/output_file.out", 'w')

list_w = [4]
list_rank = [5]

for w in list_w:
    print("w:", w)
    f.write('w: ' + str(w) +'\n\n')
    
    periodicity = 4
    X_real = X_original.copy()
    for i in range(n):
        X_real[i, d0:d] = X_original[i, d0-periods_to_forecast:d0]
    X_real = X_real.flatten(order='C')
    
    # %%
    
    b=d/(periodicity*w)
    row = int(b*n)
    
    X_original_block = np.split(X_real, np.shape(X_real)[0]/periodicity)    
    X_original_block_2 = np.zeros((row, w*periodicity))
    tau_2 = np.zeros(row)
    
    jindex = 0
    a = 0
    for index in range(len(X_original_block)):
        c = X_original_block[index]
        X_original_block_2[jindex, a:a+periodicity] = c
        a += periodicity
        if a == w*periodicity:
            jindex += 1
            a = 0
    
    index2 = b-1
    for jndex1 in range(int(row/b)):
        index2 = int(index2)
        tau_2[index2] = 1
        index2 += b
    
    tau = np.argmax(X_original_block_2 != 0, axis=1)
    
    tau_1a = tau_2[(~np.all(X_original_block_2 == 0, axis=1)) | (tau_2==1)]
    X_original_block_1 = X_original_block_2[(~np.all(X_original_block_2 == 0, axis=1)) | (tau_2==1)] 
    
    # %%
        
    m = np.shape(X_original_block_1)[0]
    
    for p in list_rank:
        np.random.seed(0)
    
        s = np.random.uniform(size=(m, p - 1))
        s = np.sort(s, axis=1)
        k = np.ones(m) - s[:, -1]
        j = np.diag(np.ones(p - 2), k=1)
        s = s - np.dot(s, j)
        W_init = np.c_[s, k]
        check = np.sum(W_init, axis=1)
        
        H_init = np.random.rand(p, w*periodicity)*100
    
        start_time = time.time()
    
        print("rank:", p)
    
        M_HALS, H_HALS, W_HALS, objective_HALS = amf.mnmf(p, X_original_block_1, H_init, W_init, tau_1a, periods_to_forecast, max_iter=5000)
        print('Error: %5.4f' % np.linalg.norm(M_HALS - np.dot(W_HALS, H_HALS)))
        print("elapsed time: %5.2f seconds" % (time.time() - start_time))
    
        save_fileobj = ('mask_test/objective_hals_rank_' + str(w) + '_' + str(p) +'.csv')
        save_fileM = ('mask_test/matrix_hals-m_rank_' + str(w) + '_' + str(p) +'.csv')
        save_fileH = ('mask_test/matrix_hals-h_rank_' + str(w) + '_' + str(p) +'.csv')
        save_fileW = ('mask_test/matrix_hals-w_rank_' + str(w) + '_' + str(p) +'.csv')
    
        np.savetxt(save_fileobj, objective_HALS, delimiter=',')
        np.savetxt(save_fileM, M_HALS, delimiter=',')
        np.savetxt(save_fileH, H_HALS, delimiter=',')
        np.savetxt(save_fileW, W_HALS, delimiter=',')
    
        matrix = pd.read_csv('mask_test/matrix_hals-w_rank_' + str(w) + '_' + str(p) +'.csv', delimiter=',', header=None)
        W_HALS = matrix.values
    
        matrix = pd.read_csv('mask_test/matrix_hals-h_rank_' + str(w) + '_' + str(p) +'.csv', delimiter=',', header=None)
        H_HALS = matrix.values
    
        matrix = pd.read_csv('mask_test/matrix_hals-m_rank_' + str(w) + '_' + str(p) +'.csv', delimiter=',', header=None)
        M_HALS = matrix.values
        
        M_new_block = np.zeros((np.shape(X_original_block_2)[0], w*periodicity))
        M_new_block[(~np.all(X_original_block_2 == 0, axis=1)) | (tau_2==1)] = M_HALS[:]
        M_new = np.zeros((n, d))
    
        jindex = 0
        a = 0
        for index in range(np.shape(X_original_block_2)[0]):
            c = M_new_block[index, :]
            M_new[jindex, a:a+periodicity*w] = c
            a += periodicity*w
            if a == d:
                jindex += 1
                a = 0
                
        M_new = M_new[:, d0:d] 
        
        prevision_error_HALS_rrmse = amf.rrmse(X_test, M_new)
        prevision_error_HALS_l1 = amf.l1_error(X_test, M_new)
        prevision_error_HALS_mpe = amf.mpe(X_test, M_new)
        prevision_error_HALS_l2 = amf.l2_error(X_test, M_new)
        
        print('rrmse:', prevision_error_HALS_rrmse)
        print('mpe:', prevision_error_HALS_mpe)
        print('l2:', prevision_error_HALS_l2)
        print('l1:', prevision_error_HALS_l1)  
    
        f.write('p: ' + str(p)  + '\n')
        f.write('rrmse: ' + str(prevision_error_HALS_rrmse)  + '\n')
        f.write('mpe: ' + str(prevision_error_HALS_mpe)  + '\n')
        f.write('l2: ' + str(prevision_error_HALS_l2)  + '\n')
        f.write('l1: ' + str(prevision_error_HALS_l1)  + '\n')
        f.write('\n')   
    
        if amf.rrmse(X_test, M_new) <= best_error:
            w_best = w
            p_best = p
            X_forecast_best = M_new.copy()
            best_error = amf.rrmse(X_test, M_new)
                    
save_file_forecast_best = ('matrix_forecast_best_mask_test.csv')
np.savetxt(save_file_forecast_best, X_forecast_best, delimiter=',')
                    
print('best rank w:', w_best) 
print('best rank p:', p_best) 
print('best RRMSE:', best_error)

f.write('best rank w: ' + str(w_best) + '\n')    
f.write('best rank p: ' + str(p_best) + '\n')
f.write('best RRMSE: ' + str(best_error) + '\n')
                    
f.close()
    
# %%
    
trajectory_list = random.sample(range(n), 5)

for i in trajectory_list:
    print(amf.rrmse(X_test[i, :], X_forecast_best[i, :]))
    amf.trajectory_plot(i, X_test, X_forecast_best, p_best)
        
# %%

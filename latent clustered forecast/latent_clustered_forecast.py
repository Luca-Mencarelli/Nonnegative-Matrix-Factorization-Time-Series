#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import include_lcf as nmf
import time
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import make_scorer, recall_score, precision_score, mean_absolute_error
import random

import sys
sys.setrecursionlimit(10000)

import multiprocessing as mp

np.random.seed(0)

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

X_original = df.transpose().values
n = np.shape(X_original)[0]
d = np.shape(X_original)[1]

d0 = d-periods_to_forecast
X_real = X_original.copy()

X_train = X_original[:, :d0]
X_test = X_original[:, d0:d0+periods_to_forecast]

# %%

import os

current_directory = os.getcwd()
matrix_dir = os.path.join(current_directory, r'matrix')
archetypes_dir = os.path.join(current_directory, r'archetypes')
trajectories_dir = os.path.join(current_directory, r'trajectories')
if not os.path.exists(matrix_dir):
    os.makedirs(matrix_dir)
if not os.path.exists(archetypes_dir):
    os.makedirs(archetypes_dir)
if not os.path.exists(trajectories_dir):
    os.makedirs(trajectories_dir)

# %%

max_depth_list = [4]
time_windows_list = [4]
list_p = [4]

best_error = 1000000

f = open("output_file_calibration.out", 'w')

for p in list_p:
    
    s = np.random.uniform(size=(n, p - 1))
    s = np.sort(s, axis=1)
    k = np.ones(n) - s[:, -1]
    j = np.diag(np.ones(p - 2), k=1)
    s = s - np.dot(s, j)
    W_init = np.c_[s, k]
    check = np.sum(W_init, axis=1)
    H_init = np.random.rand(p, np.shape(X_train)[1])

    start_time = time.time()
  
    M_HALS, H_HALS, W_HALS, objective_HALS = nmf.mnmf(p, X_train, H_init, W_init, max_iter=5000)
    
    print("elapsed time: %5.2f seconds" % (time.time() - start_time))
    X_HALS = np.maximum(0, np.dot(W_HALS, H_HALS))
    print('RRMSE =', nmf.rrmse(X_train, X_HALS))
            
    save_fileobj = ('matrix/objective_hals_rank_' + str(p) + '_entire.csv')
    save_fileM = ('matrix/matrix_hals-m_rank_' + str(p) + '_entire.csv')
    save_fileH = ('matrix/matrix_hals-h_rank_' + str(p) + '_entire.csv')
    save_fileW = ('matrix/matrix_hals-w_rank_' + str(p) + '_entire.csv')
            
    np.savetxt(save_fileobj, objective_HALS, delimiter=',')
    np.savetxt(save_fileM, M_HALS, delimiter=',')
    np.savetxt(save_fileH, H_HALS, delimiter=',')
    np.savetxt(save_fileW, W_HALS, delimiter=',')
    
    matrix = pd.read_csv('matrix/matrix_hals-w_rank_' + str(p) + '_entire.csv', delimiter=',', header=None)
    W_HALS = matrix.values

    matrix = pd.read_csv('matrix/matrix_hals-h_rank_' + str(p) + '_entire.csv', delimiter=',', header=None)
    H_HALS = matrix.values

    matrix = pd.read_csv('matrix/matrix_hals-m_rank_' + str(p) + '_entire.csv', delimiter=',', header=None)
    M_HALS = matrix.values

    model = AgglomerativeClustering(compute_full_tree=True, linkage="complete", affinity='l1')
    model.fit(W_HALS)
    n_samples = n
    minimal_size = p

    if 'rtp1' in locals() or 'rtp1' in globals():
        rtp1.clear()
        
    if 'rtp2' in locals() or 'rtp2' in globals():
        rtp2.clear()
    
    rtp = []
    rtp2 = nmf.tree_explorer(model, np.size(model.children_), n, minimal_size)
    rtp1 = rtp2.copy()
    
    for i in range(len(rtp2)):
        if (rtp2[i])[0] >= n_samples:
            rtp1[i].pop(0)     
    
    number_of_clusters = len(rtp1)
    labels = np.zeros(n)
    
    index_label = 0
    for i in range(len(rtp1)):
        for j in rtp1[i]:
            labels[j] = index_label
        index_label += 1

    W_cluster_all = np.zeros((n, p, len(rtp1)))
    
    for i in range(len(rtp1)):
        for j in rtp1[i]:
            for k in range(p):
                W_cluster_all[j, k, i] = W_HALS[j, k]
    
    W_cluster_final = [None] * len(rtp1)
    
    for i in range(len(rtp1)):
        a = W_cluster_all[:, :, i]
        W_cluster_final[i] = a[~(a==0).all(1)]
    
    X_cluster_all = np.zeros((n, d0, len(rtp1)))
    
    for i in range(len(rtp1)):
        for j in rtp1[i]:
            for k in range(d0):
                X_cluster_all[j, k, i] = X_train[j, k]
                
    X_c = [None] * len(rtp1)
    
    for i in range(len(rtp1)):
        a = X_cluster_all[:, :, i]
        X_c[i] = a[~(a==0).all(1)]
   
    cluster_with_full_rank = []
    cluster_small = []
    
    for i in range(len(rtp1)):
        if len(rtp1[i]) == 1:    
            cluster_small.append(i)
        cluster_with_full_rank.append(i)

    X_c_new = [None] * len(rtp1)
    H_test = [None] * len(cluster_with_full_rank)
    W_test = [None] * len(cluster_with_full_rank)
    cluster_index = 0
      
    print('total number of clusters:', len(cluster_with_full_rank))
    print('number of processes:', mp.cpu_count())

    pool = mp.Pool(mp.cpu_count())        
    X_c_new = pool.starmap(nmf.archetypes, [(X_c[i], p, i) for i in cluster_with_full_rank])
    pool.close() 

    for cluster_index in cluster_with_full_rank:
        matrix = pd.read_csv('matrix/H_test_cluster_' + str(p) + '_' + str(cluster_index) + '_entire.csv', delimiter=',', header=None)
        H_test[cluster_index] = matrix.values
        
        matrix = pd.read_csv('matrix/W_test_cluster_' + str(p) + '_' + str(cluster_index) + '_entire.csv', delimiter=',', header=None)
        W_test[cluster_index] = matrix.values
        
        X_c_new[cluster_index] = np.dot(W_test[cluster_index], H_test[cluster_index])

    model = 'random_forest'
    cluster_index = len(cluster_with_full_rank)

    time_window_index = 0
    H_pred = [None] * cluster_index
    X_c_final = [None] * cluster_index

    for time_window in time_windows_list: 
        for max_depth in max_depth_list:
            
            max_depth_list1 = []
            max_depth_list1.append(max_depth)
            
            rdf = RandomForestRegressor(criterion='mse')
            clf = GridSearchCV(estimator=rdf, param_grid=dict(max_depth=max_depth_list1), n_jobs=-1, scoring='neg_mean_squared_error')
        
            for c in range(cluster_index):
                H_pred[c] = np.zeros((p, d-d0))
                for i in range(p):    

                    time_window = int(np.floor(time_window))
                    print('processing time window:', time_window)
                    print('processing max_depth:', max_depth)
                    print('processing archetype', i, 'of cluster', c, 'for rank', p)

                    H_random_forest = (H_test[c])[i, :d0]
                    X, y = nmf.split_sequence(H_random_forest, time_window, d-d0) 
                    clf.fit(X, y) 
		
                    H_test1 = (H_test[c])[i, d0-time_window:d0]
                    H_test1 = H_test1.reshape((1, time_window))
                    (H_pred[c])[i, :] = clf.predict(H_test1)
                    print('forecasts:', (H_pred[c])[i, :])
                    
                    save_fileH_pred = ('archetypes/matrix_hals-h_pred_' + str(time_window) + '_' + str(max_depth) + '_rank_' + str(p) + '_' + str(c) + '_' + str(i) + '_entire.csv')
                    np.savetxt(save_fileH_pred, (H_pred[c])[i, :], delimiter=',')
               
                    matrix = pd.read_csv('archetypes/matrix_hals-h_pred_' + str(time_window) + '_' + str(max_depth) + '_rank_' + str(p) + '_' + str(c) + '_' + str(i) + '_entire.csv', delimiter=',', header=None)		    
                    (H_pred[c])[i, :] = matrix.values[:, 0]
                
                X_c_final[c] = np.maximum(0, np.dot(W_test[c], H_pred[c]))
                
            X_HALS_final = np.zeros((n, d-d0))            
                    
            for i in range(len(rtp1)):
                index = 0
                for j in rtp1[i]:
                    for k in range(d-d0): 
                        X_HALS_final[j, k] = (X_c_final[i])[index, k]
                    index += 1
            
            print('rank p:', p) 
            print('max_depth:', max_depth)
            print('time_window:', time_window)
	    
            print('rrmse:', nmf.rrmse(X_test, X_HALS_final))
            print('mpe:', nmf.mpe(X_test, X_HALS_final))
            print('l2:', nmf.l2_error(X_test, X_HALS_final))
            print('l1:', nmf.l1_error(X_test, X_HALS_final))
            
            f.write('rank p: ' + str(p) + '\n')
            f.write('max_depth: ' + str(max_depth) + '\n')
            f.write('time_window: ' + str(time_window) + '\n')
            
            f.write('rrmse: ' + str(nmf.rrmse(X_test, X_HALS_final)) + '\n')
            f.write('mpe: ' + str(nmf.mpe(X_test, X_HALS_final)) + '\n')
            f.write('l2: ' + str(nmf.l2_error(X_test, X_HALS_final)) + '\n')
            f.write('l1: ' + str(nmf.l1_error(X_test, X_HALS_final)) + '\n\n')
            
            if nmf.rrmse(X_test, X_HALS_final) < best_error:
                p_best = p
                max_depth_best = max_depth
                time_window_best = time_window
                X_forecast_best = X_HALS_final.copy()
                best_error = nmf.rrmse(X_test, X_HALS_final)
                
save_file_forecast_best = ('matrix_forecast_best_entire.csv')
np.savetxt(save_file_forecast_best, X_forecast_best, delimiter=',')
                
print('best rank p:', p_best) 
print('best max_depth:', max_depth_best)
print('best time_window:', time_window_best)
print('best RRMSE:', best_error)

f.write('best rank p: ' + str(p_best) + '\n')
f.write('best max_depth: ' + str(max_depth_best) + '\n')
f.write('best time_window: ' + str(time_window_best) + '\n\n')
                
f.close()

# %%

trajectory_list = random.sample(range(n), 5)

for i in trajectory_list:
    nmf.trajectory_plot(i, X_test, X_forecast_best, p_best)
    print(nmf.mpe(X_test[i, :], X_forecast_best[i, :]))


    


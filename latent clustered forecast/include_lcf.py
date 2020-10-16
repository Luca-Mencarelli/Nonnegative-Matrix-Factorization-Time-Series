#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

def project_simplex(x):

    n = len(x)
    xord = -np.sort(-x)
    sx = np.sum(x)
    lam = (sx-1.)/n
    if (lam<=xord[n-1]):
        return (x-lam)
    k = n-1
    flag = 0
    while ((flag==0)and(k>0)):
        sx -= xord[k]
        lam = (sx-1.)/k
        if ((xord[k]<=lam)and(lam<=xord[k-1])):
            flag = 1
        k -= 1
    return np.fmax(x-lam,0)

# %%

def size(model, node, s):
    if node < s:
        return 1
    [child1, child2] = model.children_[node-s]
    return size(model, child1, s) + size(model, child2, s)


def rtn_all_children(model, node, s):
    if node >= s:
        for child in model.children_[node-s]:
            if child < s:
                rtp.append(child)
            else:
                rtn_all_children(model, child, s)
    return [node] + rtp

# %%

rtp = []
rtl = []

rtp1 = []

def tree_explorer(model, node, s, min_dimension):
    [child1, child2] = model.children_[node-s]
    if size(model, child1, s) <= min_dimension and size(model, child2, s) <= min_dimension:
            rtp1.append(rtn_all_children(model, node, s))
            rtp.clear()
    else:
        for child in model.children_[node-s]:
            # print(child)
            if size(model, child, s) <= min_dimension:
                rtp1.append(rtn_all_children(model, child, s))
                rtp.clear()
            else:
                tree_explorer(model, child, s, min_dimension)
    return rtp1

# %%

def norm(vector):
    vector = np.array(vector)
    if (np.count_nonzero(vector) == 0):
        return 0.0
    else:
        return np.sqrt(np.sum(np.power(vector,2)))

def norml1(vector):
    vector = np.array(vector)
    if (np.count_nonzero(vector) == 0):
        return 0.0
    else:
        return np.sum(np.abs(vector))
    
def rrmse(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    if (np.count_nonzero(y_true - y_pred) == 0):
        return 0.0
    else:
        return norm(y_true - y_pred) / norm(y_true)

def mpe(y_true, y_pred):
    if (y_true == y_pred).all():
        return 0.0
    else:
        return np.linalg.norm(y_true - y_pred, 1) / np.linalg.norm(y_true, 1)

def l1_error(y_true, y_pred):
    if (y_true == y_pred).all():
        return 0.0
    else:
        return np.linalg.norm(y_true - y_pred, 1) / np.size(y_true)

def l2_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    if (np.count_nonzero(y_true - y_pred) == 0):
        return 0.0
    else:
        return norm(y_true - y_pred) / np.size(y_true)

# %%

def split_sequence(sequence, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the sequence
        if out_end_ix > len(sequence):
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)    

# %%

eps = 1e-5
niter = 1000
# c1 = 1.1
# c2 = 1.1
delta = 1e-5
epsilon = 1e-6
threshold = 1e-8


def update_U_and_V(H, H_old, I, W, W_old, G_old, V_old, Y, t, t_old, X, gamma1, gamma2):

    Inew = I.copy()
    Jnew = I.copy()
    Ynew = Y.copy()
    Znew = Y.copy()

    G = H + (t_old / t) * (I - H) + ((t_old - 1) / t) * (H - H_old)
    V = W + (t_old / t) * (Y - W) + ((t_old - 1) / t) * (W - W_old)

    res = np.dot(V, G) - X[:]
    G_temp = G.copy() - np.dot(np.transpose(V), res) / gamma1

    r = G_temp.shape[0]
    Inew[:] = G_temp[:]

    res = np.dot(W, H) - X[:]
    H_temp = H.copy() - np.dot(np.transpose(W), res) / gamma1
    Jnew[:] = H_temp[:]

    res = np.dot(V, Inew) - X
    V_temp = V[:] - (1 / gamma2) * np.dot(res, np.transpose(Inew))
    n = V_temp.shape[0]

    for i in range(0, n):
        Ynew[i, :] = project_simplex(V_temp[i, :])

    res = np.dot(W, Jnew) - X
    W_temp = W[:] - (1 / gamma2) * np.dot(res, np.transpose(Jnew))

    for i in range(0, n):
        Znew[i, :] = project_simplex(W_temp[i, :])

    tnew = (np.sqrt(4 * np.power(t, 2) + 1) + 1) / 2

    cost1 = np.power(np.linalg.norm(X - np.dot(Ynew, Inew)), 2)
    cost2 = np.power(np.linalg.norm(X - np.dot(Znew, Jnew)), 2)

    if cost1 <= cost2:
        Vnew = Ynew
        Unew = Inew
    else:
        Vnew = Znew
        Unew = Jnew

    return Unew, Vnew, Inew, Ynew, G, V, tnew


def mnmff(K, X, H_init, W_init, max_iter):

    not_finish = True
    iter = 1
    objective_val = []
    U = H_init.copy()
    V = W_init.copy()
    M = np.dot(V, U)

    U_old = U.copy()
    V_old = V.copy()
    I = U.copy()
    Y = V.copy()
    t = 1.0
    t_old = 0.0
    G_old = U.copy() + (t_old / t) * (I - U) + ((t_old - 1) / t) * (U - U_old)
    W_old = V.copy() + (t_old / t) * (Y - V) + ((t_old - 1) / t) * (V - V_old)

    M = np.dot(V, U)
    Err = np.linalg.norm(X - np.dot(V, U))
    Err_perc = np.linalg.norm(X - np.dot(V, U)) / np.linalg.norm(X)
    # print('Error =', Err)
    # print('Error percentage =', Err_perc)
    
    # objective_val.append(Err)

    while not_finish:
        # print(iter)
        alpha1 = (iter - 1.0) / (iter + 1.0)
        beta1 = alpha1
        alpha2 = 0
        beta2 = alpha2
        # c1 = (1 + 2 * beta1) / (2 - 2 * alpha1)
        c1 = 1.1
        c2 = c1
        gamma1 = c1 * np.linalg.norm(np.dot(V.transpose(), V))
        gamma2 = c2 * np.linalg.norm(np.dot(U, U.transpose()))
        gamma2 = max(gamma2, 1e-4)
        Unew, Vnew, Inew, Ynew, G, W, tnew = update_U_and_V(U, U_old, I, V, V_old, G_old, W_old, Y, t, t_old, X, gamma1, gamma2)

        if (np.linalg.norm(U - Unew) <= delta) and (np.linalg.norm(V - Vnew) <= delta):
            not_finish = False
        
        U_old = U.copy()
        V_old = V.copy()
        G_old = G.copy()
        W_old = W.copy()
        t_old = t
        t = tnew
        I = Inew.copy()
        Y = Ynew.copy()
        U = Unew.copy()
        V = Vnew.copy()

        M = np.dot(V, U)
        
        Err = np.linalg.norm(X - np.dot(V, U))
        Err_perc = np.linalg.norm(X - np.dot(V, U)) / np.linalg.norm(X)
        # print('Error =', Err)
        # print('Error percentage =', Err_perc)

        objective_val.append(Err)

        if iter == max_iter:
            not_finish = False

        iter += 1

    return M, U, V, objective_val


def mnmf(K, X, H_init, W_init, max_iter=100):

    n = X.shape[0]
    d = X.shape[1]

    M, U, V, objective_val = mnmff(K, X, H_init, W_init, max_iter)
    Err = np.linalg.norm(X - np.dot(V, U))
    Err_perc = np.linalg.norm(X - np.dot(V, U)) / np.linalg.norm(X)

    return M, U, V, objective_val
   
# %%
    
def archetypes(A1, p, cluster_index):

    # H_init = nmf.initH(A1, p)
    s = np.random.uniform(size=(np.shape(A1)[0], p-1))
    s = np.sort(s, axis=1)
    k = np.ones(np.shape(A1)[0]) - s[:, -1]
    j = np.diag(np.ones(p - 2), k=1)
    s = s - np.dot(s, j)
    W_init = np.c_[s, k]
    check = np.sum(W_init, axis=1)
    H_init = np.random.rand(p, np.shape(A1)[1])
        
    _, H_test, W_test, objective_test = mnmf(p, A1, H_init, W_init, max_iter=5000)
    save_file_h_cluster = ('matrix/H_test_cluster_' + str(p) + '_' + str(cluster_index) + '_entire.csv')
    np.savetxt(save_file_h_cluster, H_test, delimiter=',')
    
    save_file_w_cluster = ('matrix/W_test_cluster_' + str(p) + '_' + str(cluster_index) + '_entire.csv')
    np.savetxt(save_file_w_cluster, W_test, delimiter=',')
    
    save_file_obj = ('matrix/objective_hals_test_rank_' + str(p) + '_' + str(cluster_index) + '_entire.csv')
    np.savetxt(save_file_obj, objective_test, delimiter=',')       
    
    B1 = np.dot(W_test, H_test)
    return B1

# %%

def trajectory_plot(trajectory, original, reconstructed, rank):
    f = plt.subplots()
    plt.plot(original[trajectory, :], label="Original")
    plt.plot(reconstructed[trajectory, :], label="Reconstructed")
    plt.legend()
    plt.savefig("trajectories/trajectory_X_%s_%s.pdf" % (rank, trajectory))
    plt.show()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.ndimage.interpolation import shift
from sklearn.metrics import mean_absolute_error

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

eps = 1e-5
niter = 1000
c1 = 1.1
c2 = c1
delta = 1e-5
epsilon = 1e-6
threshold = 1e-8

def projectM(Ma, Matrix, tau_1a, w, periodicity, periods_to_forecast):
    n1, n2 = Ma.shape
    M1 = Ma.copy()
    for i in range(n1):
        if tau_1a[i] == 0:
            M1[i, :] = Matrix[i, :]
        else:
            M1[i, :(w-1)*periodicity-periods_to_forecast] = Matrix[i, :(w-1)*periodicity-periods_to_forecast]
    return M1

def projectM1(Ma, Matrix, U, V, tau_1a, w, periodicity, periods_to_forecast, gamma3=1.1):
    M1 = Ma + 1 / gamma3 * (np.dot(V, U) - Ma)
    n1, n2 = Ma.shape
    M2 = M1.copy()
    for i in range(n1):
        if tau_1a[i] == 0:
            M2[i, :] = Matrix[i, :]
        else:
            M2[i, :(w-1)*periodicity-periods_to_forecast] = Matrix[i, :(w-1)*periodicity-periods_to_forecast]
    return M2


def update_U_and_V(H, H_old, I, W, W_old, G_old, V_old, Y, t, t_old, X, gamma1, gamma2):
    
    Inew = I.copy()
    Jnew = I.copy()
    Ynew = Y.copy()
    Znew = Y.copy()

    G = H + (t_old / t) * (I - H) + ((t_old - 1) / t) * (H - H_old)
    V = W + (t_old / t) * (Y - W) + ((t_old - 1) / t) * (W - W_old)

    res = np.dot(V, G) - X[:]
    G_temp = G.copy() - np.dot(np.transpose(V), res) / gamma1

    r = np.maximum(0, G_temp.shape[0])
    Inew[:] = G_temp[:] 

    res = np.dot(W, H) - X[:]
    H_temp = H.copy() - np.dot(np.transpose(W), res) / gamma1
    Jnew[:] = np.maximum(0, H_temp[:])

    res = np.dot(V, Inew) - X
    V_temp = V[:] - (1 / gamma2) * np.dot(res, np.transpose(Inew))
    
    for i in range(0, np.shape(V_temp)[0]):
        Ynew[i, :] = project_simplex(V_temp[i, :])

    res = np.dot(W, Jnew) - X
    W_temp = W[:] - (1 / gamma2) * np.dot(res, np.transpose(Jnew))
    
    for i in range(0, np.shape(V_temp)[0]):
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


def mnmff(K, X, H_init, W_init, max_iter, tau_1a, w, periodicity, periods_to_forecast):

    not_finish = True
    iter = 1
    prevision_best = 1000
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

    M = projectM(M, X, tau_1a, w, periodicity, periods_to_forecast)

    Err = np.linalg.norm(M - np.dot(V, U))
    Err_perc = np.linalg.norm(M - np.dot(V, U)) / np.linalg.norm(M)
    
    # print('Error =', Err)
    # print('Error percentage =', Err_perc)
    # objective_val.append(Err)

    while not_finish:
        
        # print(iter)
        obj = np.dot(V, U) - M

        gamma1 = c1 * np.linalg.norm(np.dot(V.transpose(), V))
        gamma2 = c2 * np.linalg.norm(np.dot(U, U.transpose()))
        gamma2 = max(gamma2, 1e-4)
        
        M_old = M.copy()
        Unew, Vnew, Inew, Ynew, G, W, tnew = update_U_and_V(U, U_old, I, V, V_old, G_old, W_old, Y, t, t_old, M, gamma1, gamma2)
        
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
        
        # M = projectM(np.dot(V, U), X, tau_1a, periods_to_forecast)
        M = projectM1(M_old, X, U, V, tau_1a, w, periodicity, periods_to_forecast)
         
        Err = np.linalg.norm(M - np.dot(V, U))
        Err_perc = np.linalg.norm(M - np.dot(V, U)) / np.linalg.norm(M)
        # print('Error =', Err)
        # print('Error percentage =', Err_perc)

        objective_val.append(Err)

        if iter == max_iter:
            not_finish = False

        iter += 1

    return M, U, V, objective_val


def mnmf(K, X, H_init, W_init, tau_1a, w, periodicity, periods_to_forecast, max_iter=100):

    nC = X.shape[0]
    dC = X.shape[1]

    M, U, V, objective_val = mnmff(K, X, H_init, W_init, max_iter, tau_1a, w, periodicity, periods_to_forecast)
    Err = np.linalg.norm(M - np.dot(V, U))
    Err_perc = np.linalg.norm(M - np.dot(V, U)) / np.linalg.norm(M)
    # print('Error =', Err)
    # print('Error percentage =', Err_perc)

    return M, U, V, objective_val

# %%
    
def trajectory_plot(trajectory, original, reconstructed, rank):
    f = plt.subplots()
    plt.plot(original[trajectory, :], label="Original")
    plt.plot(reconstructed[trajectory, :], label="Reconstructed")
    plt.legend()
    plt.savefig("trajectories/trajectory_X_%s_%s.pdf" % (rank, trajectory))
    plt.show()
    
# %%

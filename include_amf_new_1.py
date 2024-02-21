#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 17 18:06:36 2020

@author: luca mencarelli
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import NMF as nmf
import time

np.random.seed(0)

# %%
    
def rrmse(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    if (np.count_nonzero(y_true - y_pred) == 0):
        return 0.0
    else:
        return np.linalg.norm(y_true - y_pred, 2) / np.linalg.norm(y_true, 2)

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
        return np.linalg.norm(y_true - y_pred, 2) / np.size(y_true)
        
# %%

eps = 1e-5
niter = 1000
c1 = 1.1
c2 = c1
delta = 1e-5
epsilon = 1e-6
threshold = 1e-8

alpha1 = 1
beta1 = 1
alpha2 = 1
beta2 = 1

tol=10e-4

def projectM(Ma, Matrix, t1, t2, tau_1a,periods_to_forecast):
    n1, n2 = Ma.shape
    M1 = Ma.copy()
    M1 = np.maximum(0, Ma)
    for i in range(n1):
        if tau_1a[i] == 0:
            M1[i, :] = Matrix[i, :]
        else:
            M1[i, :n2-periods_to_forecast] = Matrix[i, :n2-periods_to_forecast]
    return M1

def projectM1(Ma, Matrix, t1, t2, U, V, periods_to_forecast,tau_1a, gamma3=1.1):
    M1 = Ma + 1 / gamma3 * (np.dot(V, U) - Ma)
    M1 = np.maximum(0, M1)
    n1, n2 = Ma.shape
    M2 = M1.copy()
    for i in range(n1):
        if tau_1a[i] == 0:
            M2[i, :] = Matrix[i, :]
        else:
            M2[i, :n2-periods_to_forecast] = Matrix[i, :n2-periods_to_forecast]
    return M2


def update_U_and_V(H, H_old, I, W, W_old, G_old, V_old, Y, t, t_old, X, l, gamma1, gamma2, alpha1, beta1, alpha2, beta2):
    Inew = I.copy()
    Jnew = I.copy()
    Ynew = Y.copy()
    Znew = Y.copy()

    G = H + (t_old / t) * (I - H) + ((t_old - 1) / t) * (H - H_old)
    V = W + (t_old / t) * (Y - W) + ((t_old - 1) / t) * (W - W_old)

    res = np.dot(V, G) - X[:]
    G_temp = G.copy() - np.dot(np.transpose(V), res) / gamma1

    r = G_temp.shape[0] 

    for i in range(0, r):
        #G_grad, _ = nmf.wolfe_proj(X - G_temp[i, :], epsilon=epsilon, threshold=threshold, niter=niter)   
        Inew[i, :] = G_temp[i, :] #+ (l / (l + gamma1)) * G_grad    
    #Inew = np.maximum(0, Inew)

    res = np.dot(W, H) - X[:]
    H_temp = H.copy() - np.dot(np.transpose(W), res) / gamma1

    for i in range(0, r):
        #H_grad, _ = nmf.wolfe_proj(X - H_temp[i, :], epsilon=epsilon, threshold=threshold, niter=niter)
        Jnew[i, :] = H_temp[i, :] #+ (l / (l + gamma1)) * H_grad
    #Jnew = np.maximum(0, Jnew)

    res = np.dot(V, Inew) - X
    V_temp = V[:] - (1 / gamma2) * np.dot(res, np.transpose(Inew))
    
    for i in range(0, np.shape(V_temp)[0]):
        Ynew[i, :] = nmf.project_simplex(V_temp[i, :])

    res = np.dot(W, Jnew) - X
    W_temp = W[:] - (1 / gamma2) * np.dot(res, np.transpose(Jnew))
    
    for i in range(0, np.shape(V_temp)[0]):
        Znew[i, :] = nmf.project_simplex(W_temp[i, :])
    
    tnew = (np.sqrt(4 * np.power(t, 2) + 1) + 1) / 2

    cost1 = np.power(np.linalg.norm(X - np.dot(Ynew,Inew)),2)
    cost2 = np.power(np.linalg.norm(X - np.dot(Znew,Jnew)),2)

    if cost1 <= cost2:
        Vnew = Ynew
        Unew = Inew
    else:
        Vnew = Znew
        Unew = Jnew

    return Unew, Vnew, Inew, Ynew, G, V, tnew


def mnmff(K, X, H_init, W_init, penalty, max_iter, t_initial, t_final,  d0, X_original_block_1,X_original_block_2,periodicity,tau_2,n,d,w,tau_1a,periods_to_forecast,X_test):
    
    not_finish = True
    iter = 1
    prevision_best = 1000
    objective_val = []
    # N, T = X.shape
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

    M = projectM(M, X_original_block_1, t_initial, t_final,tau_1a,periods_to_forecast)

    Err = np.linalg.norm(M - np.dot(V, U))
    Err_perc = np.linalg.norm(M - np.dot(V, U)) / np.linalg.norm(M)
    
    #print('Error =', Err)
    #print('Error percentage =', Err_perc)
    '''
    M_new_block = np.zeros((np.shape(X_original_block_2)[0], w*periodicity))
    M_new_block[(~np.all(X_original_block_2 == 0, axis=1)) | (tau_2==1)] = M[:]
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
    
    prevision_error_HALS_rrmse = rrmse(X_test, M_new)
    prevision_error_HALS_l1 = l1_error(X_test, M_new)
    prevision_error_HALS_mpe = mpe(X_test, M_new)
    prevision_error_HALS_l2 = l2_error(X_test, M_new)
    
    #print('rrmse:', prevision_error_HALS_rrmse)
    #print('mpe:', prevision_error_HALS_mpe)
    #print('l2:', prevision_error_HALS_l2)
    #print('l1:', prevision_error_HALS_l1)  
    
    # objective_val.append(Err)
    '''
    while not_finish:
        
        #print(iter)
        obj = np.dot(V, U) - M

        gamma1 = c1 * np.linalg.norm(np.dot(V.transpose(), V))
        gamma2 = c2 * np.linalg.norm(np.dot(U, U.transpose()))
        gamma2 = max(gamma2, 1e-4)
        
        M_old = M.copy()
        Unew, Vnew, Inew, Ynew, G, W, tnew = update_U_and_V(U, U_old, I, V, V_old, G_old, W_old, Y, t, t_old, M, penalty, gamma1, gamma2, alpha1, beta1, alpha2, beta2)
        
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
        
        #M = projectM(np.dot(V, U), X_original_block_1, t_initial, t_final,tau_1a,periods_to_forecast)
        M = projectM1(M_old, X_original_block_1, t_initial, t_final, U, V,periods_to_forecast,tau_1a,)

        '''
        if Err < np.linalg.norm(X - np.dot(V, U)):
            print("ERROR")
            break
        '''
        if np.abs(np.linalg.norm(M - np.dot(V, U)) / np.linalg.norm(M)-Err_perc) < tol:
            not_finish = False
 
        obj = np.dot(V, U) - M
        n1, n2 = np.shape(V)
        n3, n4 = np.shape(U)

        RW = np.zeros((n1, n2))
        RH = np.zeros((n3, n4))

        obj1 = np.dot(obj, U.transpose())
        obj2 = np.dot(V.transpose(), obj)

        for i in range(n1):
            for j in range(n2):
                if V[i, j] != 0:
                    RW[i, j] = obj1[i, j]

        for i in range(n3):
            for j in range(n4):
                if U[i, j] != 0:
                    RH[i, j] = obj2[i, j]

        c = np.linalg.norm(RW) + np.linalg.norm(RH)
        #print(c)
	
        if c <= 10e-7:
            not_finish = False
        
        Err = np.linalg.norm(M - np.dot(V, U))
        Err_perc = np.linalg.norm(M - np.dot(V, U)) / np.linalg.norm(M)
        #print('Error =', Err)
        #print('Error percentage =', Err_perc)
        
        M_new_block = np.zeros((np.shape(X_original_block_2)[0], w*periodicity))
        M_new_block[(~np.all(X_original_block_2 == 0, axis=1)) | (tau_2==1)] = M[:]
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
        
        prevision_error_HALS_rrmse = rrmse(X_test, M_new)
        prevision_error_HALS_l1 = l1_error(X_test, M_new)
        prevision_error_HALS_mpe = mpe(X_test, M_new)
        prevision_error_HALS_l2 = l2_error(X_test, M_new)
        
        #print('rrmse:', prevision_error_HALS_rrmse)
        #print('mpe:', prevision_error_HALS_mpe)
        #print('l2:', prevision_error_HALS_l2)
        #print('l1:', prevision_error_HALS_l1)  
            
        # print(M[0, d0-tau[0]:d])
        # print(M_new[0, 0:d0])
        # print(M[0, :])

        objective_val.append(Err)

        # if Err_perc <= eps:
        #    not_finish = False

        if iter == max_iter:
            not_finish = False

        iter += 1

    return M, U, V, objective_val


def mnmf(K, X, H_init, W_init, t_initial, t_final, X_original_block_1,X_original_block_2,periodicity,tau_2,n,d,d0,w,tau_1a,periods_to_forecast, max_iter, X_test,lmin=0.001, lmax=10, lambda_no=20, c_lambda=1.2):
    # lambdas = np.geomspace(lmin, lmax, lambda_no)
    lambdas = np.linspace(lmin, lmax, lambda_no)
    l_no = 0

    nC = X.shape[0]
    dC = X.shape[1]

    conv_hull_loss = 0

    if lambda_no > 1:
        if (dC <= K):
            pca_loss = 0
        else:
            proj_X = nmf.project_principal(X, K)
            pca_loss = np.linalg.norm(X - proj_X)

    for i in range(lambda_no):
        #print('lambda =', lambdas[i])
        
        M, U, V, objective_val = mnmff(K, X, H_init, W_init, lambdas[i], max_iter, t_initial, t_final, d0, X_original_block_1,X_original_block_2,periodicity,tau_2,n,d,w,tau_1a,periods_to_forecast,X_test)
        Err = np.linalg.norm(M - np.dot(V, U))
        Err_perc = np.linalg.norm(M - np.dot(V, U)) / np.linalg.norm(M)
        # print('Error =', Err)
        # print('Error percentage =', Err_perc)
        #print(' ')

        if lambda_no > 1:
            for j in range(0, nC):
                projXj, _ = nmf.wolfe_proj(U - X[j, :], epsilon=epsilon, threshold=threshold, niter=niter)
                conv_hull_loss = conv_hull_loss + (np.power(np.linalg.norm(projXj), 2))

            conv_hull_loss = np.sqrt(conv_hull_loss)

            l_lambda = conv_hull_loss - pca_loss
            if (l_no == 0):
                l_lambda0 = l_lambda
            if (l_lambda >= l_lambda0 * c_lambda):
                break

        l_no += 1
    return M, U, V, objective_val

# %%
    
def trajectory_plot(trajectory, original, reconstructed, rank):
    f = plt.subplots()
    plt.plot(original[trajectory, :], label="Original")
    plt.plot(reconstructed[trajectory, :], label="Reconstructed")
    plt.legend()
    plt.savefig("trajectory_X_%s_%s.pdf" % (rank, trajectory))
    # plt.show()

# %%

def experiments_amf(X_original,periods_to_forecast,list_w,list_rank,periodicity,log_file):
    
    n = np.shape(X_original)[0]
    d = np.shape(X_original)[1]
    d0 = d-periods_to_forecast

    X_train = X_original[:, :d0]
    X_test = X_original[:, d0:d0+periods_to_forecast] 

    X_original = X_original[~np.all(X_train==0, axis=1)]
    X_train = X_original[:, :d0]
    X_test = X_original[:, d0:d0+periods_to_forecast]
    n = np.shape(X_original)[0]
    d = np.shape(X_original)[1]

    h = np.max(X_original[:,:-periods_to_forecast])

    best_error_rrmse = 1000000
    best_error_mpe = 1000000
    f = open(log_file, 'w')

    for w in list_w:
        
        X_real = X_original.copy()
        # X_real[:, d0:d] = 0
        for i in range(n):
            X_real[i, d0:d] = X_original[i, d0-periods_to_forecast:d0]
        X_real = X_real.flatten(order='C')
        
        # %%
        
        b=d/(periodicity*w)
        # print((b-w+1)*n)
        #row = int((b-w+1)*n)
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
        '''        
        M_new_block = np.zeros((np.shape(X_original_block_2)[0], w*periodicity))
        M_new_block[(~np.all(X_original_block_2 == 0, axis=1)) | (tau_2==1)] =  X_original_block_1
        
        #M_new_block[:] = X_original_block_2[:]
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
        
        prevision_error_HALS_rrmse = rrmse(X_test, M_new)
        prevision_error_HALS_l1 = l1_error(X_test, M_new)
        prevision_error_HALS_mpe = mpe(X_test, M_new)
        prevision_error_HALS_l2 = l2_error(X_test, M_new)
        
        #print('rrmse:', prevision_error_HALS_rrmse)
        #print('mpe:', prevision_error_HALS_mpe)
        #print('l2:', prevision_error_HALS_l2)
        #print('l1:', prevision_error_HALS_l1)  
        
        # %%
        '''    
        tinit = np.zeros(np.shape(X_original_block_1)[0])
        tfin = np.zeros(np.shape(X_original_block_1)[0])
        m = np.shape(X_original_block_1)[0]
        
        # int(0.2*np.floor(np.shape(X_original_block_1)[0]))
        
        for p in list_rank:
            np.random.seed(0)

            start_time = time.time()
        
            #H_init = nmf.initH(X_original_block_1, p)
            s = np.random.uniform(size=(m, p - 1))
            s = np.sort(s, axis=1)
            k = np.ones(m) - s[:, -1]
            j = np.diag(np.ones(p - 2), k=1)
            s = s - np.dot(s, j)
            W_init = np.c_[s, k]
            #W_init = np.ones((m, p))/p
            check = np.sum(W_init, axis=1)
            #print(check)
            
            H_init = np.random.rand(p, w*periodicity)*h
            #H_init = np.dot(np.linalg.pinv(W_init), X_original_block_1)
            #W_init = np.random.rand(m, p)
        
            start_time = time.time()
        
            #print("w:", w)
            #print("rank:", p)
        
            M_HALS, H_HALS, W_HALS, objective_HALS = mnmf(p, X_original_block_1, H_init, W_init, tinit, tfin, X_original_block_1,X_original_block_2,periodicity,tau_2,n,d,d0,w,tau_1a,periods_to_forecast, 5000, X_test,lmin=0, lmax=0, lambda_no=1, c_lambda=1.2)
            #print('Error HALS on semiNMF: %5.4f' % np.linalg.norm(M_HALS - np.dot(W_HALS, H_HALS)))
            #print(' ')
        
            
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
            
            prevision_error_HALS_rrmse = rrmse(X_test, M_new)
            prevision_error_HALS_l1 = l1_error(X_test, M_new)
            prevision_error_HALS_mpe = mpe(X_test, M_new)
            prevision_error_HALS_l2 = l2_error(X_test, M_new)

            elapsed_time = time.time() - start_time
            #print("elapsed time: ", elapsed_time)
            
            #print('rrmse:', prevision_error_HALS_rrmse)
            #print('mpe:', prevision_error_HALS_mpe)
            #print('l2:', prevision_error_HALS_l2)
            #print('l1:', prevision_error_HALS_l1)  
        
            f.write('w: ' + str(w) +'\n')
            f.write('p: ' + str(p)  + '\n')
            f.write('rrmse: ' + str(prevision_error_HALS_rrmse)  + '\n')
            f.write('mpe: ' + str(prevision_error_HALS_mpe)  + '\n')
            f.write('l2: ' + str(prevision_error_HALS_l2)  + '\n')
            f.write('l1: ' + str(prevision_error_HALS_l1)  + '\n')
            f.write('time: ' + str(elapsed_time)  + '\n')
            f.write('\n')   
        
            if rrmse(X_test, M_new) <= best_error_rrmse:
                w_best_rrmse = w
                p_best_rrmse = p
                #X_forecast_best = M_new.copy()
                best_error_rrmse = rrmse(X_test, M_new)
                elapsed_time_rrmse = elapsed_time

            if mpe(X_test, M_new) <= best_error_mpe:
                w_best_mpe = w
                p_best_mpe = p
                #X_forecast_best = M_new.copy()
                best_error_mpe = mpe(X_test, M_new)
                elapsed_time_mpe = elapsed_time
                        
    #save_file_forecast_best = ('matrix_forecast_best_mask_test.csv')
    #np.savetxt(save_file_forecast_best, X_forecast_best, delimiter=',')

    f.write('best rank w: ' + str(w_best_rrmse) + '\n')    
    f.write('best rank p: ' + str(p_best_rrmse) + '\n')
    f.write('best RRMSE: ' + str(best_error_rrmse) + '\n')

    f.write('best rank w: ' + str(w_best_mpe) + '\n')    
    f.write('best rank p: ' + str(p_best_mpe) + '\n')
    f.write('best MPE: ' + str(best_error_mpe) + '\n')
                        
    f.close()

    return w_best_rrmse, p_best_rrmse, best_error_rrmse, elapsed_time_rrmse, w_best_mpe, p_best_mpe, best_error_mpe, elapsed_time_mpe
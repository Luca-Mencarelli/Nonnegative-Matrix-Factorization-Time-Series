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

tol = 10e-4

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


def mnmff(K, X_original_block_1, H_init, W_init, t_initial, t_final, X_original_block_2,periodicity,tau_2,n,d,d0,w,tau_1a,periods_to_forecast, max_iter, X_test):
    not_finish = True
    iter = 1
    prevision_best = 1000
    objective_val = []
    # N, T = X.shape
    U = H_init.copy()
    V = W_init.copy()
    
    alpha = 0.00

    M = np.dot(V, U)

    U_old = U.copy()
    U_new = U.copy()
    V_old = V.copy()
    V_new = V.copy()

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
    rho_H = 1 + n*(d+K)/(d*(K+1))
    rho_W = 1 + d*(n+K)/(n*(K+1))

    while not_finish:
        
        #print(iter)
        obj = np.dot(V, U) - M

        for k in range(int(np.floor(1+alpha*rho_W))):
            A = np.dot(M,np.transpose(U_new))
            B = np.dot(U_new,np.transpose(U_new))
            V_new = V_old.copy()

            for ell in range(K):
                C1=0
                C2=0
                for j in range(ell):
                    C1 += V_new[:,j] * B[j,ell]
                for j in range(ell+1,K):
                    C2 += V_old[:,j] * B[j,ell]
                C = C1 + C2
                if B[ell,ell]!=0.0:
                    V_new[:,ell] = np.maximum(0,(A[:,ell]-C[:])/B[ell,ell])
                else:
                    V_new[:,ell]=0
                AT = nmf.project_simplex(np.transpose(V_new[:,ell]))
                V_new[:,ell] = np.transpose(AT)
            V_old = V_new.copy()
        V_old = V_new.copy()
        U_old = U_new.copy()

        M = projectM(np.dot(V_new, U_new), X_original_block_1, t_initial, t_final,tau_1a,periods_to_forecast)

        for k in range(int(np.floor(1+alpha*rho_H))):
            A = np.dot(V_new.transpose(),M)
            B = np.dot(np.transpose(V_new),V_new)
            #U_new = np.zeros((np.shape(U_old)[0],np.shape(U_old)[1]))

            for ell in range(K):
                C1=0
                C2=0
                for j in range(ell):
                    C1 += U_new[j,:] * B[ell,j]
                for j in range(ell+1,K):
                    C2 += U_old[j,:] * B[ell,j]
                C = C1 + C2
                if B[ell,ell]!=0.0:
                    U_new[ell,:] = np.maximum(0,(A[ell,:]-C[:])/B[ell,ell])
                else:
                    U_new[ell,:] = 0
            U_old = U_new.copy()
        
        #M = projectM(np.dot(V, U), X_original_block_1, t_initial, t_final)
        U = U_new.copy()
        V = V_new.copy()
        #M = projectM1(np.dot(V, U), X_original_block_1, t_initial, t_final, U_new, V_new,periods_to_forecast,tau_1a)
        M = projectM(np.dot(V, U), X_original_block_1, t_initial, t_final,tau_1a,periods_to_forecast)
        
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
	
        if c <= 10e-2:
            not_finish = False
        
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

        print(prevision_error_HALS_rrmse)
        
        #print('rrmse:', prevision_error_HALS_rrmse)
        #print('mpe:', prevision_error_HALS_mpe)
        #print('l2:', prevision_error_HALS_l2)
        #print('l1:', prevision_error_HALS_l1)  
            
        # print(M[0, d0-tau[0]:d])
        # print(M_new[0, 0:d0])
        # print(M[0, :])
        '''
        objective_val.append(Err)

        if Err_perc <= eps:
            not_finish = False

        if iter == max_iter:
            not_finish = False

        iter += 1

    return M, U, V, objective_val

def trajectory_plot(trajectory, original, reconstructed, rank):
    f = plt.subplots()
    plt.plot(original[trajectory, :], label="Original")
    plt.plot(reconstructed[trajectory, :], label="Reconstructed")
    plt.legend()
    plt.savefig("trajectory_X_%s_%s.pdf" % (rank, trajectory))
    # plt.show()

# %%

def experiments_nmf(X_original,periods_to_forecast,list_w,list_rank,periodicity,log_file):
    
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
        row = int(b*n*2)
        size = int(periodicity)
        step = int(periodicity/2)
        
        X_original_block = [X_real[i : i + size] for i in range(0, len(X_real), step)]
        X_original_block_2 = np.zeros((row, w*periodicity))
        tau_2 = np.zeros(row)

        pp = int(periodicity/2)
        
        jindex = 0
        a = 0
        for index in range(len(X_original_block)-1):
            c = X_original_block[index]
            X_original_block_2[jindex, a:a+periodicity] = c
            a += periodicity
            if a == w*periodicity:
                jindex += 1
                a = 0
        
        index2 = b*2-1
        for jndex1 in range(int(row/b/2)):
            index2 = int(index2)
            tau_2[index2] = 1
            index2 += b*2

        tau = np.argmax(X_original_block_2 != 0, axis=1)    
        tau_1a = tau_2[(~np.all(X_original_block_2 == 0, axis=1)) | (tau_2==1)]
        X_original_block_1 = X_original_block_2[(~np.all(X_original_block_2 == 0, axis=1)) | (tau_2==1)]
        
        # %%
            
        tinit = np.zeros(np.shape(X_original_block_1)[0])
        tfin = np.zeros(np.shape(X_original_block_1)[0])
        m = np.shape(X_original_block_1)[0]
        
        # int(0.2*np.floor(np.shape(X_original_block_1)[0]))
        
        for p in list_rank:
            np.random.seed(0)

            start_time = time.time()
        
            # H_init = nmf.initH(X_original_block_1, p)
            s = np.random.uniform(size=(m, p - 1))
            s = np.sort(s, axis=1)
            k = np.ones(m) - s[:, -1]
            j = np.diag(np.ones(p - 2), k=1)
            s = s - np.dot(s, j)
            W_init = np.c_[s, k]
            # W_init = np.ones((m, p))/p
            check = np.sum(W_init, axis=1)
            
            H_init = np.random.rand(p, w*periodicity)*h
            # H_init = np.dot(np.linalg.pinv(W_init), X_original_block_1)
            # W_init = np.random.rand(m, p)
        
            #print("w:", w)
            #print("rank:", p)
        
            M_HALS, H_HALS, W_HALS, objective_HALS = mnmff(p, X_original_block_1, H_init, W_init, tinit, tfin, X_original_block_2,periodicity,tau_2,n,d,d0,w,tau_1a,periods_to_forecast, 5000, X_test)
            #print('Error HALS on semiNMF: %5.4f' % np.linalg.norm(M_HALS - np.dot(W_HALS, H_HALS)))
            #print(' ')
            
            M_new_block = np.zeros((np.shape(X_original_block_2)[0], w*periodicity))
            M_new_block[(~np.all(X_original_block_2 == 0, axis=1)) | (tau_2==1)] =  M_HALS[:]
            M_new = np.zeros((n, d-d0))
            
            jindex = 0
            
            for i in range(np.shape(M_new_block)[0]):
                if tau_2[i] == 1:
                    M_new[jindex, :] = M_new_block[i, w*periodicity-periods_to_forecast:w*periodicity]
                    jindex += 1
            
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

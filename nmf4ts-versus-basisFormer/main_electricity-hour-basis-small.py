#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from tabulate import tabulate
import os
import random

import include_amf_new_1_overlap as amf_overlap
import include_nmf_new_1_overlap as nmf_overlap
import include_amf_new_1 as amf
import include_nmf_new_1 as nmf
import include_benchmark as bmrk

from sklearn import preprocessing

# %%

'''Data pre-processing'''

df = pd.read_csv('electricity.csv') 
print(df)

df.index = pd.to_datetime(df["date"], format='%Y-%m-%d %H:%M:%S')
df1 = df.groupby(pd.Grouper(freq="H")).sum()

for i in range(10):
    df = df1.iloc[960*i:960*(i+1), :]
    df.to_csv('out_electricity_'+str(i)+'.csv')
    #df = df.transpose()
    print(df)

    #x = df.values
    #min_max_scaler = preprocessing.MinMaxScaler()
    #x_scaled = min_max_scaler.fit_transform(x)
    #df = pd.DataFrame(x_scaled)

    name = "electricity_basis_"+str(i)+"_"

    # %%

    periods_to_forecast = 24

    X_original = df.transpose().values

    list_w1 = [2,4,5,10]
    list_rank1 = [5, 10, 20, 30, 40, 50]
    periodicity1 = 24

    list_w2 = [2,4,5,10]
    list_rank2 = [5, 10, 20, 30, 40, 50]
    periodicity2 = 2*periodicity1

    w_best_rrmse_AMF, p_best_rrmse_AMF, best_error_rrmse_AMF, elapsed_time_rrmse_AMF, w_best_mpe_AMF, p_best_mpe_AMF, best_error_mpe_AMF, elapsed_time_mpe_AMF = amf.experiments_amf(X_original,periods_to_forecast,list_w1,list_rank1,periodicity1,name+"log_file_AMF")
    w_best_rrmse_NMF, p_best_rrmse_NMF, best_error_rrmse_NMF, elapsed_time_rrmse_NMF, w_best_mpe_NMF, p_best_mpe_NMF, best_error_mpe_NMF, elapsed_time_mpe_NMF = nmf.experiments_nmf(X_original,periods_to_forecast,list_w1,list_rank1,periodicity1,name+"log_file_NMF")
    w_best_rrmse_AMF_overlap, p_best_rrmse_AMF_overlap, best_error_rrmse_AMF_overlap, elapsed_time_rrmse_AMF_overlap, w_best_mpe_AMF_overlap, p_best_mpe_AMF_overlap, best_error_mpe_AMF_overlap, elapsed_time_mpe_AMF_overlap = amf_overlap.experiments_amf(X_original,periods_to_forecast,list_w2,list_rank2,periodicity2,name+"log_file_AMF_overlap")
    w_best_rrmse_NMF_overlap, p_best_rrmse_NMF_overlap, best_error_rrmse_NMF_overlap, elapsed_time_rrmse_NMF_overlap, w_best_mpe_NMF_overlap, p_best_mpe_NMF_overlap, best_error_mpe_NMF_overlap, elapsed_time_mpe_NMF_overlap = nmf_overlap.experiments_nmf(X_original,periods_to_forecast,list_w2,list_rank2,periodicity2,name+"log_file_NMF_overlap")

    '''
    #elapsed_time, error_rrmse, error_mpe 
    elapsed_timeRFR, error_rrmseRFR, error_mpeRFR = bmrk.experiments_rfr(X_original,periods_to_forecast,periodicity1)
    elapsed_timeLSTM, error_rrmseLSTM, error_mpeLSTM = bmrk.experiments_neural_network_LSTM(X_original,periods_to_forecast,periodicity1)
    elapsed_timeGRU, error_rrmseGRU, error_mpeGRU = bmrk.experiments_neural_network_GRU(X_original,periods_to_forecast,periodicity1)
    elapsed_timeEXP, error_rrmseEXP, error_mpeEXP = bmrk.experiments_exponential_smoothing(X_original,periods_to_forecast,periodicity1)
    elapsed_timeSARIMAX, error_rrmseSARIMAX, error_mpeSARIMAX = bmrk.experiments_SARIMAX(X_original,periods_to_forecast,periodicity1)
    '''
    RESULT = [[] for i in range(5)]

    RESULT[0].append("MODEL")
    RESULT[0].append("RRMSE")
    RESULT[0].append("TIME")
    RESULT[0].append("RMPE")
    RESULT[0].append("TIME")

    RESULT[1].append("AMF")
    RESULT[1].append(best_error_rrmse_AMF)
    RESULT[1].append(elapsed_time_rrmse_AMF)
    RESULT[1].append(best_error_mpe_AMF)
    RESULT[1].append(elapsed_time_mpe_AMF)

    RESULT[2].append("AMF OVERLAP")
    RESULT[2].append(best_error_rrmse_AMF_overlap)
    RESULT[2].append(elapsed_time_rrmse_AMF_overlap)
    RESULT[2].append(best_error_mpe_AMF_overlap)
    RESULT[2].append(elapsed_time_mpe_AMF_overlap)

    RESULT[3].append("NMF")
    RESULT[3].append(best_error_rrmse_NMF)
    RESULT[3].append(elapsed_time_rrmse_NMF)
    RESULT[3].append(best_error_mpe_NMF)
    RESULT[3].append(elapsed_time_mpe_NMF)

    RESULT[4].append("NMF OVERLAP")
    RESULT[4].append(best_error_rrmse_NMF_overlap)
    RESULT[4].append(elapsed_time_rrmse_NMF_overlap)
    RESULT[4].append(best_error_mpe_NMF_overlap)
    RESULT[4].append(elapsed_time_mpe_NMF_overlap)

    '''
    RESULT[5].append("RFR")
    RESULT[5].append(error_rrmseRFR)
    RESULT[5].append(elapsed_timeRFR)
    RESULT[5].append(error_mpeRFR)
    RESULT[5].append(elapsed_timeRFR)

    RESULT[6].append("LSTM")
    RESULT[6].append(error_rrmseLSTM)
    RESULT[6].append(elapsed_timeLSTM)
    RESULT[6].append(error_mpeLSTM)
    RESULT[6].append(elapsed_timeLSTM)

    RESULT[7].append("GRU")
    RESULT[7].append(error_rrmseGRU)
    RESULT[7].append(elapsed_timeGRU)
    RESULT[7].append(error_mpeGRU)
    RESULT[7].append(elapsed_timeGRU)

    RESULT[8].append("EXP")
    RESULT[8].append(error_rrmseEXP)
    RESULT[8].append(elapsed_timeEXP)
    RESULT[8].append(error_mpeEXP)
    RESULT[8].append(elapsed_timeEXP)

    RESULT[9].append("SARIMAX")
    RESULT[9].append(error_rrmseSARIMAX)
    RESULT[9].append(elapsed_timeSARIMAX)
    RESULT[9].append(error_mpeSARIMAX)
    RESULT[9].append(elapsed_timeSARIMAX)
    '''
    f = open(name+'results.out', 'w') 
    f.write(tabulate(RESULT, headers='firstrow'))
    f.write('\n')
    f.close()
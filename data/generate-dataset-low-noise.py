#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 17 18:06:36 2020

@author: luca mencarelli
"""

import pandas as pd
import numpy as np
import random

np.random.seed(0)

# %%

'''Data pre-processing'''
periodicity = 10
number_series = 1000
number_periods = 10
sigma = 0.005

series = np.zeros((number_series, periodicity))
dataset = np.zeros((number_series, periodicity*number_periods))

for i in range(number_series):
    series[i] = np.random.rand(periodicity)
    
dataset = np.tile(series, number_periods)
#print(dataset)

for i in range(number_series):
    for j in range(periodicity*number_periods):
        dataset[i, j] += sigma * random.random()
        
#print(dataset)
        
df = pd.DataFrame(data=dataset)
print(df)

df.to_csv('dataset_file-low-noise.csv', index = False, header=True)




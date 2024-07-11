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
df = df.groupby(pd.Grouper(freq="H")).sum()

df = df.iloc[:960, :]
#df = df.transpose()
print(df)

df.to_csv('out.csv')


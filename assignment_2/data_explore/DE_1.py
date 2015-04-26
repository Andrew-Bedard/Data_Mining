# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 16:54:50 2015

@author: Andrew Bedard
"""

import pandas as pd
import numpy as np
import numpy.random as rnd

rnd.seed(10)

"""Removes duplicate values from seq"""
def f7(seq):
    seen = set()
    seen_add = seen.add
    return [ x for x in seq if not (x in seen or seen_add(x)) ]


pct = 0.15 #Aprox. percentage of origional csv file to randomly select

""" Load data set a few variables"""
data = pd.read_csv('training_set_VU_DM_2014.csv')
col_list = list(data.columns.values)
index = range(int(pct*len(data)))
length = len(data)

""" Create random array of integers based on pct, then use f7() to remove duplicates"""
t1 = rnd.randint(0,length,int(pct*length))   
t2 = f7(t1)

"""Data frame for the fraction of values we will consider"""
df = pd.DataFrame(index=t2, columns=col_list)

"""Copy data from data to dataframe"""
for i in range(len(t2)):
    R = rnd.random()
    if R <= pct:
        df[i:i+1] = data[t2[i]:t2[i+1]]
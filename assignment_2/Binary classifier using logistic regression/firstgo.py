# -*- coding: utf-8 -*-
"""
Created on Tue May 12 13:22:18 2015

@author: Andy
"""

import pandas as pd
import numpy as np
import numpy.random as rnd

""" Load data set """
data = pd.read_csv('decimated_training_set_10_pct.csv')
data = data.drop('date_time',1)
col_list = list(data.columns.values)

""" Normalised Price """

#Remove outliers
data = data[data.price_usd <= 5000]
length = len(data)
#To speed up computation, assign dummmy variales to columns then compute
dummy_price = data['price_usd']
dummy_length = data['srch_length_of_stay']
data['ppn_usd'] = dummy_price/dummy_length

""" Separating Countries by id"""

gb = data.groupby(['prop_country_id'])
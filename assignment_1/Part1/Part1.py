# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 00:13:56 2015

@author: Andy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import sklearn as sk

data = pd.read_csv('C:/Users/Andy/Documents/School/Data_Mining/Assignment_1/ODI3.csv')
col_list = list(data.columns.values)

columns = ['Prog','ML','IR','Db','Sex','Chocolate','Neighb','Stress','Money','Random','Bed']
index = range(len(data))

df = pd.DataFrame(index=index, columns = columns)


##df['Prog'] = data[col_list[0]].map( {'bioinformatics': 1, 'econometrics':2, 'cs':3, 'ba':4, 'ai':5, 'computational science':3})
#Set sex values of df to 0 for female, 1 for male
df['Sex'] = data[col_list[5]].map( {'f': -1, 'm': 1} ).astype(int)
#Set Machine Learning (ML) values of df to 0 for no, 1 for yes
df['ML'] = data[col_list[1]].map( {'n': -1, 'y': 1} ).astype(int)
#Set Chocolate values of df to 1 for fat, -1 for slim, 0 for neither, and 99 for I have no idea
df['Chocolate'] = data[col_list[6]].map( {'fat': 1, 'slim': -1, 'neither': 0, 'I have no idea': 2} ).astype(float)
df['IR'] = data[col_list[2]]
#Set Database (Db) values of df to 1 for j, 0 for n
df['Db'] = data[col_list[4]].map({'j': 1, 'n': -1}).astype(int)
df['Neighb'] = data[col_list[8]]
df['Stress'] = data[col_list[10]]
#Someone put something crazy for their answer in Money, so I had to force pandas
#to try conversion to numeric, and if not, set as NaN
df['Money'] = data[col_list[11]].convert_objects(convert_numeric=True)
df['Random'] = data[col_list[12]]
#df['Bed'] = data[col_list[13]].convert_objects(convert_dates=True)
#It is unreasonable that someone would have more than the maximum of the Moore Neighbourhood
#(8 neighbours) thus rows with values greater than 8 are removed, because now I dont trust that slippery devil
df = df[df.Neighb < 9]
#Same thing as before, someone cant follow directions, so all their data is removed!!!
df = df[df.Money <= 100]


fig, ax = plt.subplots()

a_heights, a_bins = np.histogram(df['Sex'])
b_heights, b_bins = np.histogram(df['ML'], bins=a_bins)
width = (a_bins[1] - a_bins[0])/3
ax.bar(a_bins[:-1], a_heights, width=width, facecolor='cornflowerblue')
ax.bar(b_bins[:-1]+width, b_heights, width=width, facecolor='seagreen')
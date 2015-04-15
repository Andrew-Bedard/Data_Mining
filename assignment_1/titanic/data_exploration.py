import pandas
import matplotlib as plt
from matplotlib import pyplot

df = pandas.read_csv('train.csv')
fig = pyplot.figure()
ax = fig.add_subplot(111)
ax.hist(df['Age'].dropna(), bins = 30, range = (df['Age'].min(),df['Age'].max()))
pyplot.xlabel("age")
pyplot.ylabel("number of passengers")
pyplot.savefig('../report/age_distribution.png')

ax.cla()
survivors_ages = df[df.Survived==1]['Age'].dropna().values
ax.hist(survivors_ages, bins = 30, range = (df['Age'].min(),df['Age'].max()))
pyplot.xlabel("age")
pyplot.ylabel("number of passengers")
pyplot.savefig('../report/survivor_age_distribution.png')


ax.cla()
non_survivors_ages = df[df.Survived==0]['Age'].dropna().values
ax.hist(non_survivors_ages, bins = 30, range = (df['Age'].min(),df['Age'].max()))
pyplot.xlabel("age")
pyplot.ylabel("number of passengers")
pyplot.savefig('../report/non_survivor_age_distribution.png')

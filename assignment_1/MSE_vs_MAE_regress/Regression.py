# Data imported from http://vincentarelbundock.github.io/Rdatasets/csv/datasets/longley.csv

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error

df = pd.read_csv('longley.csv', index_col=0)


y = df.Employed  # dependent variable
df['squared_GNP']= df.GNP**2
#X = df.GNP
X = df[['GNP','squared_GNP']] #predictor
# X = df[['GNP','squared_GNP','Population']]

X = sm.add_constant(X)  # Adds a constant term to the predictor

est = sm.OLS(y, X)

est = est.fit()
est.summary()
est.params

y_hat = est.predict(X)

errors = y - y_hat
MSE = mean_squared_error(y, y_hat)

MAE = np.mean(np.abs(errors))

print(est.params)
print(errors)
print(MSE,MAE)



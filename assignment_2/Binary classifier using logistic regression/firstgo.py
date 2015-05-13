# -*- coding: utf-8 -*-
"""
Created on Tue May 12 13:22:18 2015

@author: Andy
"""

import pandas as pd
import numpy as np
from patsy import dmatrices
import sklearn as sk
from sklearn.linear_model import LogisticRegression

""" Load data set """
data = pd.read_csv('decimated_training_set_10_pct.csv')
data = data.drop('date_time',1)
data = data.drop(data.columns[26:50], 1)


""" Adding and congragating features """

#Normalised Price

#Remove price outliers
data = data[data.price_usd <= 5000]
length = len(data)
#To speed up computation, assign dummmy variales to columns then compute
dummy_price = data['price_usd']
dummy_length = data['srch_length_of_stay']
data['ppn_usd'] = dummy_price/dummy_length


#ump = exp(prop_log_historical_price) - price_usd
dummy_histp = np.exp(data['prop_log_historical_price'])
data['ump'] = dummy_histp - dummy_price


#price_diff = visitor_hist_adr_usd - price_usd
dummy_hist_ad = data['visitor_hist_adr_usd']
data['price_diff'] = dummy_hist_ad - dummy_price


#starrating_diff = visitor_hist_starrating - prop_starrating
dummy_hstar = data['visitor_hist_starrating']
dummy_pstar = data['prop_starrating']
data['starrating_diff'] = dummy_hstar - dummy_pstar



""" Filling in Nan values of prop_location_score2 with the first quantile """
 
f = lambda x: x.fillna(x.quantile(0.25))
data.prop_location_score2 = data.groupby('prop_country_id')["prop_location_score2"].transform(f)




col_list = list(data.columns.values)

""" Down sample negative instances """

"""Lets try some stuff"""

y, X = dmatrices('click_bool ~ visitor_hist_adr_usd + prop_country_id + \
prop_id + prop_starrating + prop_review_score + prop_brand_bool + \
prop_location_score1 + prop_location_score2 + prop_log_historical_price + \
position + promotion_flag + srch_destination_id + srch_length_of_stay + \
srch_booking_window + srch_adults_count + srch_children_count + \
srch_room_count + srch_saturday_night_bool + srch_query_affinity_score + \
orig_destination_distance + random_bool + ppn_usd + ump + price_diff + \
starrating_diff', data, return_type="dataframe")

print(X.columns)

y = np.ravel(y)

model = LogisticRegression(class_weight = 'auto')
model = model.fit(X,y)

model.score(X,y)
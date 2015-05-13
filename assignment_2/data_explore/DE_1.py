# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 16:54:50 2015

@author: Andrew Bedard
"""

import pandas as pd
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

rnd.seed(10)

data = pd.read_csv('training_set_VU_DM_2014.csv')
data = data.drop('date_time',1)
datasize = len(data)

col_list = list(data.columns.values)

desc = data.describe()
df = desc.copy()

sum_list = []

for i in range(len(col_list)):
    dummy = df[col_list[i]]
    sum_list.append(dummy[0])
    
    
diff = [np.abs(sum_list[i] - sum_list[0]) for i in range(len(col_list))]
diff_pct = np.zeros((1,len(col_list)))

for i in range(len(col_list)):
    if diff[i] != 0:
        diff_pct[0,i] = (diff[i]/sum_list[0])
        
pos = np.arange(len(sum_list))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.barh(pos, np.array(diff_pct*100), align='center', height=0.1)
ax.set_yticks(pos, ('srch_id','site_id','visitor_location_country_id','visitor_hist_starrating','visitor_hist_adr_usd','prop_country_id','prop_id','prop_starrating','prop_review_score','prop_brand_bool','prop_location_score1','prop_location_score2','prop_log_historical_price','position','price_usd','promotion_flag','srch_destination_id','srch_length_of_stay','srch_booking_window','srch_adults_count','srch_children_count','srch_room_count','srch_saturday_night_bool','srch_query_affinity_score','orig_destination_distance','random_bool','comp1_rate','comp1_inv','comp1_rate_percent_diff','comp2_rate','comp2_inv','comp2_rate_percent_diff','comp3_rate','comp3_inv','comp3_rate_percent_diff','comp4_rate','comp4_inv','comp4_rate_percent_diff','comp5_rate','comp5_inv','comp5_rate_percent_diff','comp6_rate','comp6_inv','comp6_rate_percent_diff','comp7_rate','comp7_inv','comp7_rate_percent_diff','comp8_rate','comp8_inv','comp8_rate_percent_diff','click_bool','gross_bookings_usd','booking_bool'))

ax.axvline(0, color='k', lw=3)

ax.set_xlabel('Percent missing')
ax.set_title('Percentage of data missing')
ax.gird(True)
plt.show()

plt.barh()
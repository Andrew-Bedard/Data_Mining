# -*- coding: utf-8 -*-
"""
Created on Tue May 12 13:22:18 2015

@author: Andy
"""

import pandas as pd
import numpy as np
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn import metrics

""" Load data set """
data = pd.read_csv('decimated_training_set_10_pct.csv')
data = data.drop('date_time',1)
#ata = data.drop(data.columns[26:50], 1)


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
dummy_fill = data.visitor_hist_starrating.mean()
data.visitor_hist_starrating = data.visitor_hist_starrating.fillna(dummy_fill)
dummy_hstar = data['visitor_hist_starrating']
dummy_pstar = data['prop_starrating']
data['starrating_diff'] = dummy_hstar - dummy_pstar



""" Filling in Nan values of prop_location_score2 with the first quantile """
 
f = lambda x: x.fillna(x.quantile(0.25))
data.prop_location_score2 = data.groupby('prop_country_id')["prop_location_score2"].transform(f)


col_list = list(data.columns.values)

grouped = data.groupby('prop_country_id')


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

# evaluate the model by splitting into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
model2 = LogisticRegression(class_weight = 'auto')
model2.fit(X_train, y_train)

# predict class labels for the test set
predicted = model2.predict(X_test)
print(predicted)

# generate class probabilities
probs = model2.predict_proba(X_test)
print(probs)

# generate evaluation metrics
print( metrics.accuracy_score(y_test, predicted))
print( metrics.roc_auc_score(y_test, probs[:, 1]))

print( metrics.confusion_matrix(y_test, predicted))
print( metrics.classification_report(y_test, predicted))

# evaluate the model using 10-fold cross-validation
scores = cross_val_score(LogisticRegression(class_weight = 'auto'), X, y, scoring='accuracy', cv=10)
print(scores)
print(scores.mean())


#ref_df = data[['srch_id', 'booking_bool', 'click_bool']]
#ref_df['real_score'] = (ref_df.booking_bool * 5  + ref_df.click_bool)
#ref_df['predict_booking'] = booking_predictions
#ref_df['predict_click'] = click_predictions
#ref_df['predict_score'] = ref_df.predict_booking * 5 + ref_df.predict_click
#grouped = ref_df.groupby('srch_id')
#
#ndcgs = []
#for name, group in grouped:
#    real_sorted = group.sort('real_score', ascending=False)
#    idcg = 0
#    for (i, (index,val)) in enumerate(real_sorted.iterrows(), start=1):
#        idcg += (2**val.real_score-1)/numpy.log2(i+1)
#    predict_sorted = group.sort('predict_score', ascending=False)
#    dcg = 0
#    for (i, (index,val)) in enumerate(predict_sorted.iterrows(), start=1):
#        dcg += (2**val.real_score-1)/np.log2(i+1)
#    ndcgs.append(dcg/idcg)
#
#print(np.mean(ndcgs))
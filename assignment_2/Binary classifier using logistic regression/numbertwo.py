# -*- coding: utf-8 -*-


import pandas
import numpy
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn import metrics

""" Loaddf set """
df = pandas.read_csv('decimated_training_set_10_pct.csv')
df =df.drop('date_time',1)
#df =df.dropdf.columns[26:50], 1)

vals_and_stuff = ['visitor_hist_adr_usd', 'prop_country_id', 'prop_id', 'prop_starrating', \
'prop_review_score', 'prop_brand_bool', 'prop_location_score1', \
'prop_location_score2', 'prop_log_historical_price', 'position', 'promotion_flag', \
'srch_destination_id', 'srch_length_of_stay', 'srch_booking_window', \
'srch_adults_count', 'srch_children_count', 'srch_room_count', \
'srch_saturday_night_bool', 'srch_query_affinity_score',\
'orig_destination_distance', 'random_bool', 'ppn_usd', 'ump', 'price_diff', 'starrating_diff']

""" Adding and congragating features """

#Normalised Price

#Remove price outliers
df = df[df.price_usd <= 5000]
length = len(df)
#To speed up computation, assign dummmy variales to columns then compute
dummy_price =df['price_usd']
dummy_length =df['srch_length_of_stay']
df['ppn_usd'] = dummy_price/dummy_length


#ump = exp(prop_log_historical_price) - price_usd
dummy_histp = numpy.exp(df['prop_log_historical_price'])
df['ump'] = dummy_histp - dummy_price


#price_diff = visitor_hist_adr_usd - price_usd
dummy_fill = df.visitor_hist_adr_usd.mean()
df.visitor_hist_adr_usd = df.visitor_hist_adr_usd.fillna(dummy_fill)
dummy_hist_ad =df['visitor_hist_adr_usd']
df['price_diff'] = dummy_hist_ad - dummy_price


#starrating_diff = visitor_hist_starrating - prop_starrating
dummy_fill =df.visitor_hist_starrating.mean()
df.visitor_hist_starrating =df.visitor_hist_starrating.fillna(dummy_fill)
dummy_hstar =df['visitor_hist_starrating']
dummy_pstar =df['prop_starrating']
df['starrating_diff'] = dummy_hstar - dummy_pstar



""" Filling in Nan values of prop_location_score2 with the first quantile """
 
f = lambda x: x.fillna(x.mean())
df.prop_location_score2 =df.groupby('prop_country_id')["prop_location_score2"].transform(f)

col_list = list(df.columns.values)

""" Getting clickers and bookers and stuff"""
clickers = df[df.click_bool == 1]
non_clickers = df[df.click_bool == 0][:len(clickers)]
new_frame = pandas.concat([clickers,non_clickers])


bookers = df[df.booking_bool == 1]
non_clickers = df[df.booking_bool == 0][:len(bookers)]
new_frame = pandas.concat

""""""

""""""
"""Lets try some stuff"""

#Clicking Classifier

y, X = dmatrices('click_bool ~ visitor_hist_adr_usd + prop_country_id + \
prop_id + prop_starrating + prop_review_score + prop_brand_bool + \
prop_location_score1 + prop_location_score2 + prop_log_historical_price + \
position + promotion_flag + srch_destination_id + srch_length_of_stay + \
srch_booking_window + srch_adults_count + srch_children_count + \
srch_room_count + srch_saturday_night_bool + srch_query_affinity_score + \
orig_destination_distance + random_bool + ppn_usd + ump + price_diff + \
starrating_diff',df, return_type="dataframe")

y = numpy.ravel(y)

clicking_classifier = LogisticRegression(class_weight = 'auto')
clicking_classifier = clicking_classifier.fit_transform(X,y)
clicking_classifier.score(X,y)

#Booking Classifier

y, X = dmatrices('booking_bool ~ visitor_hist_adr_usd + prop_country_id + \
prop_id + prop_starrating + prop_review_score + prop_brand_bool + \
prop_location_score1 + prop_location_score2 + prop_log_historical_price + \
position + promotion_flag + srch_destination_id + srch_length_of_stay + \
srch_booking_window + srch_adults_count + srch_children_count + \
srch_room_count + srch_saturday_night_bool + srch_query_affinity_score + \
orig_destination_distance + random_bool + ppn_usd + ump + price_diff + \
starrating_diff',df, return_type="dataframe")

y = numpy.ravel(y)

booking_classifier = LogisticRegression(class_weight = 'auto')
booking_classifier = booking_classifier.fit_transform(X,y)
booking_classifier.score(X,y)

full_df = pandas.read_csv('decimated_training_set_1_pct.csv')
full_df = full_df.drop('date_time',1)

"""BEWARE, THIS WILL BE A MESS"""
"""!!!!!!!!!!!!!!!!!!!!!!!!!!!"""
""""""
#Remove price outliers
full_df = full_df[full_df.price_usd <= 5000]
length = len(full_df)
#To speed up computation, assign dummmy variales to columns then compute
dummy_price =full_df['price_usd']
dummy_length =full_df['srch_length_of_stay']
full_df['ppn_usd'] = dummy_price/dummy_length


#ump = exp(prop_log_historical_price) - price_usd
dummy_histp = numpy.exp(full_df['prop_log_historical_price'])
full_df['ump'] = dummy_histp - dummy_price


#price_diff = visitor_hist_adr_usd - price_usd
dummy_fill = full_df.visitor_hist_adr_usd.mean()
full_df.visitor_hist_adr_usd = full_df.visitor_hist_adr_usd.fillna(dummy_fill)
dummy_hist_ad =full_df['visitor_hist_adr_usd']
full_df['price_diff'] = dummy_hist_ad - dummy_price


#starrating_diff = visitor_hist_starrating - prop_starrating
dummy_fill = full_df.visitor_hist_starrating.mean()
full_df.visitor_hist_starrating = full_df.visitor_hist_starrating.fillna(dummy_fill)
dummy_hstar = full_df['visitor_hist_starrating']
dummy_pstar = full_df['prop_starrating']
full_df['starrating_diff'] = dummy_hstar - dummy_pstar

""" Filling in Nan values of prop_location_score2 with the first quantile """
 
f = lambda x: x.fillna(x.mean())
full_df.prop_location_score2 = full_df.groupby('prop_country_id')["prop_location_score2"].transform(f)

test_set = full_df[vals_and_stuff]
test_set = test_set.fillna(test_set.mean())
test_set['Intercept'] = 0
#clicking_classifier.transform(test_set)
#booking_classifier.transform(test_set)
click_predictions = clicking_classifier.predict(test_set)
booking_predictions = booking_classifier.predict(test_set)


ref_df = full_df[['srch_id', 'booking_bool', 'click_bool']]
ref_df['real_score'] = (ref_df.booking_bool * 5  + ref_df.click_bool)
ref_df['predict_booking'] = booking_predictions
ref_df['predict_click'] = click_predictions
ref_df['predict_score'] = ref_df.predict_booking * 5 + ref_df.predict_click
grouped = ref_df.groupby('srch_id')


ndcgs = []
for name, group in grouped:
    real_sorted = group.sort('real_score', ascending=False)
    idcg = 0
    for (i, (index,val)) in enumerate(real_sorted.iterrows(), start=1):
        idcg += (2**val.real_score-1)/numpy.log2(i+1)
    predict_sorted = group.sort('predict_score', ascending=False)
    dcg = 0
    for (i, (index,val)) in enumerate(predict_sorted.iterrows(), start=1):
        dcg += (2**val.real_score-1)/numpy.log2(i+1)
    ndcgs.append(dcg/idcg)

print(numpy.mean(ndcgs))
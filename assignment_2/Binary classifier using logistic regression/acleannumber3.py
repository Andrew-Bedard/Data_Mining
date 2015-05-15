# -*- coding: utf-8 -*-


import pandas
import numpy
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn import metrics


def pre_proc(DF):

	#Remove price outliers
	DF = DF[DF.price_usd <= 5000]

	#ppn_usd is price per night
	DF['ppn_usd'] = DF.price_usd/DF.srch_length_of_stay

	#ump = exp(prop_log_historical_price) - price_usd
	DF['ump'] = numpy.exp(DF.prop_log_historical_price) - DF.price_usd

	""" Filling in Nan values of numerical values with mean values within group
	where DF is grouped by prop_country_id """

	f = lambda x: x.fillna(x.mean())
	groups = DF.groupby('prop_country_id')
	DF.prop_location_score2 = groups.prop_location_score2.transform(f)
	DF.visitor_hist_adr_usd = groups.visitor_hist_adr_usd.transform(f)
	DF.visitor_hist_starrating = groups.visitor_hist_starrating.transform(f)

	#
	DF['price_diff'] = DF.visitor_hist_adr_usd - DF.price_usd

	#
	DF['starrating_diff'] = DF.visitor_hist_starrating - DF.prop_starrating

	return(DF)

def dmatrices_thingy(DF, ):

	y, X = dmatrices(' %s ~ visitor_hist_adr_usd + prop_country_id + \
	prop_id + prop_starrating + prop_review_score + prop_brand_bool + \
	prop_location_score1 + prop_location_score2 + prop_log_historical_price + \
	position + promotion_flag + srch_destination_id + srch_length_of_stay + \
	srch_booking_window + srch_adults_count + srch_children_count + \
	srch_room_count + srch_saturday_night_bool + srch_query_affinity_score + \
	orig_destination_distance + random_bool + ppn_usd + ump + price_diff + \
	starrating_diff' %  ,DF, return_type="dataframe")

""" Load Dataset """
df = pandas.read_csv('decimated_training_set_10_pct.csv')
df = df.drop('date_time',1)

vals_and_stuff = ['visitor_hist_adr_usd', 'prop_country_id', 'prop_id', 'prop_starrating', \
'prop_review_score', 'prop_brand_bool', 'prop_location_score1', \
'prop_location_score2', 'prop_log_historical_price', 'position', 'promotion_flag', \
'srch_destination_id', 'srch_length_of_stay', 'srch_booking_window', \
'srch_adults_count', 'srch_children_count', 'srch_room_count', \
'srch_saturday_night_bool', 'srch_query_affinity_score',\
'orig_destination_distance', 'random_bool', 'ppn_usd', 'ump', 'price_diff', 'starrating_diff']


df = pre_proc(df)


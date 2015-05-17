# -*- coding: utf-8 -*-


import pandas
import numpy
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn import metrics


def pre_proc(DF):

	##Replace all price_usd outliers with 999999, to be replaced with 3rd quartile values below
	#DF.price_usd = [DF.price_usd <= 5000]
	high_list = [i for i, e in enumerate(DF.price_usd) if e>=5001]
	low_list = [i for i, e in enumerate(DF.price_usd) if e <=10]

	DF.price_usd[high_list] = 999999
	DF.price_usd[low_list] = 999999


	#ppn_usd is price per night
	DF['ppn_usd'] = DF.price_usd/DF.srch_length_of_stay

	#ump = exp(prop_log_historical_price) - price_usd
	DF['ump'] = numpy.exp(DF.prop_log_historical_price) - DF.price_usd

	""" Filling in Nan values of numerical values with mean values within group
	where DF is grouped by prop_country_id """

	f = lambda x: x.fillna(x.mean())
	g = lambda x: x.replace(999999, x.quantile(0.75))
	groups = DF.groupby('prop_country_id')
	DF.prop_location_score2 = groups.prop_location_score2.transform(f)
	DF.visitor_hist_adr_usd = groups.visitor_hist_adr_usd.transform(f)
	DF.visitor_hist_starrating = groups.visitor_hist_starrating.transform(f)
	#for some queries the price_usd = 999999, replace these entries with 3rd quartile
	DF.price_usd = groups.price_usd(g)


	#
	DF['price_diff'] = DF.visitor_hist_adr_usd - DF.price_usd

	#
	DF['starrating_diff'] = DF.visitor_hist_starrating - DF.prop_starrating



	return(DF)

def dmatrices_thingy(DF, bool_var):

	y, X = dmatrices(' %s ~ visitor_hist_adr_usd + prop_country_id + \
	prop_id + prop_starrating + prop_review_score + prop_brand_bool + \
	prop_location_score1 + prop_location_score2 + prop_log_historical_price + \
	position + promotion_flag + srch_destination_id + srch_length_of_stay + \
	srch_booking_window + srch_adults_count + srch_children_count + \
	srch_room_count + srch_saturday_night_bool + srch_query_affinity_score + \
	orig_destination_distance + random_bool + ppn_usd + ump + price_diff + \
	starrating_diff' % bool_var ,DF, return_type="dataframe")

	y = numpy.ravel(y)

	return y, X


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

y, X = dmatrices_thingy(df, click_bool)

clicking_classifier = LogisticRegression(class_weight = 'auto')
clicking_classifier = clicking_classifier.fit_transform(X,y)
clicking_classifier.score(X,y)

y, X = dmatrices_thingy(df, booking_bool)

booking_classifier = LogisticRegression(class_weight = 'auto')
booking_classifier = booking_classifier.fit_transform(X,y)
booking_classifier.score(X,y)

full_df = pandas.read_csv('decimated_training_set_1_pct.csv')
full_df = full_df.drop('date_time',1)

full_df = pre_proc(full_df)

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
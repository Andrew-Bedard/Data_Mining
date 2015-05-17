# -*- coding: utf-8 -*-

import copy
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
    

    f = lambda x: x.fillna(x.mean())
    g = lambda x: x.replace(999999, x.quantile(0.25))
    h = lambda x: x.replace(0, x.quantile(0.75))
    j = lambda x: x.replace(0, x.quantile(0.25))
    k = lambda x: x.replace(0, x.mean())
    groups = DF.groupby('prop_country_id')
    #filling in Nan values
    DF.prop_location_score2 = groups.prop_location_score2.transform(f)
    DF.visitor_hist_adr_usd = groups.visitor_hist_adr_usd.transform(f)
    DF.visitor_hist_starrating = groups.visitor_hist_starrating.transform(f)
    DF.prop_starrating = groups.prop_starrating.transform(f)
    DF.prop_review_score = groups.prop_review_score.transform(f)
    #for some queries the price_usd = 999999, replace these entries with 3rd quartile
    DF.price_usd = groups.price_usd.transform(g)
    #some hotels have zero star rating, ie: no rating, replace with first quartile
    DF.prop_starrating = groups.prop_starrating.transform(h)
    DF.prop_review_score = groups.prop_review_score.transform(h)
    
    DF.prop_log_historical_price = groups.prop_log_historical_price.transform(j)
    DF.srch_length_of_stay = groups.srch_length_of_stay.transform(k)

    #ppn_usd is price per night
    DF['ppn_usd'] = DF.price_usd/DF.srch_length_of_stay

    #ump = exp(prop_log_historical_price) - price_usd
    DF['ump'] = numpy.exp(DF.prop_log_historical_price) - DF.price_usd


    #
    DF['price_diff'] = DF.visitor_hist_adr_usd - DF.price_usd
    
    #
    DF['starrating_diff'] = DF.visitor_hist_starrating - DF.prop_starrating
    
    #
    DF['per_fee'] = (DF.price_usd*DF.srch_room_count)/(DF.srch_adults_count + DF.srch_children_count)
    
    #
    DF['score1d2'] = (DF.prop_location_score2 + 0.0001)/(DF.prop_location_score1 + 0.0001)
    
    #
    DF['total_fee'] = DF.price_usd*DF.srch_room_count
    
    DF = DF.fillna(0)

    return(DF)


def dmatrices_thingy(DF, bool_var):

    y, X = dmatrices(' %s ~ prop_location_score2 + ump + price_diff +\
     starrating_diff + score1d2 + random_bool + per_fee + price_usd +\
     prop_review_score + total_fee + prop_starrating' %bool_var ,DF, return_type="dataframe")

    y = numpy.ravel(y)

    return(y, X)


""" Load Dataset """
df = pandas.read_csv('decimated_training_set_10_pct.csv')
df = df.drop('date_time',1)

print(df.visitor_hist_starrating.describe())

vals_and_stuff = ['prop_location_score2', 'ump', 'price_diff', 'starrating_diff',\
'score1d2', 'random_bool', 'per_fee', 'price_usd', 'prop_review_score',\
'total_fee', 'prop_starrating']

bool_vars = ['click_bool', 'booking_bool']

df = pre_proc(df)

print(df.visitor_hist_starrating.describe())

y, X = dmatrices_thingy(df, bool_vars[0])

clicking_classifier = LogisticRegression(class_weight = 'auto', C = 1.5)
clicking_classifier = clicking_classifier.fit(X,y)
print(clicking_classifier.score(X,y))

y, X = dmatrices_thingy(df, bool_vars[1])

booking_classifier = LogisticRegression(class_weight = 'auto', C = 1.5)
booking_classifier = booking_classifier.fit(X,y)
print(booking_classifier.score(X,y))

full_df = pandas.read_csv('decimated_training_set_1_pct.csv')
full_df = full_df.drop('date_time',1)

full_df = pre_proc(full_df)

test_set = full_df[vals_and_stuff]
test_set = test_set.fillna(test_set.mean())
#Logistic regression creates intercept colum, so we need to make one for test set
test_set['Intercept'] = 0
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
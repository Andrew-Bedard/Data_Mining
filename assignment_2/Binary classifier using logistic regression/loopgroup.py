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
    

    f = lambda x: x.fillna(x.mean())
    g = lambda x: x.replace(999999, x.quantile(0.75))
    h = lambda x: x.replace(0, x.quantile(0.25))
    groups = DF.groupby('prop_country_id')
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


    #
    DF['price_diff'] = DF.visitor_hist_adr_usd - DF.price_usd
    DF['starrating_diff'] = DF.visitor_hist_starrating - DF.prop_starrating
    DF['per_fee'] = (DF.price_usd*DF.srch_room_count)/(DF.srch_adults_count + DF.srch_children_count)
    DF['score1d2'] = (DF.prop_location_score2 + 0.0001)/(DF.prop_location_score1 + 0.0001)
    DF['total_fee'] = DF.price_usd*DF.srch_room_count

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

vals_and_stuff = ['prop_location_score2', 'ump', 'price_diff', 'starrating_diff',\
'score1d2', 'random_bool', 'per_fee', 'price_usd', 'prop_review_score',\
'total_fee', 'prop_starrating', 'prop_country_id']

bool_vars = ['click_bool', 'booking_bool']

df = pre_proc(df)


full_df = pandas.read_csv('decimated_training_set_1_pct.csv')
full_df = full_df.drop('date_time',1)
full_df = pre_proc(full_df)

test_set = full_df[vals_and_stuff]
test_set = test_set.fillna(test_set.mean())
#Logistic regression creates intercept colum, so we need to make one for test set
test_set['Intercept'] = 0


"""
Fix meeeeeeeeee 8===D
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
"""

click_predictions = []
booking_predictions = []

groups1 = df.groupby('prop_country_id')
test_group = test_set.groupby('prop_country_id')

for name, group in groups1:
    y, X = dmatrices_thingy(group, bool_vars[0])
    clicking_classifier = LogisticRegression(class_weight = 'auto')
    clicking_classifier = clicking_classifier.fit(X,y)
    q, P = dmatrices_thingy(group, bool_vars[1])
    booking_classifier = LogisticRegression(class_weight = 'auto')
    booking_classifier = booking_classifier.fit(P,q)
    
    c_pred = clicking_classifier.predict(test_group.get_group(name))
    b_pred = booking_classifier.predict(test_group.get_group(name))
    
    click_predictions.extend(c_pred)
    booking_predictions.exten(b_pred)
       






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
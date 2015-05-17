import pandas as pd 
import numpy as np
import math 
from sklearn import linear_model, cross_validation, svm, tree, naive_bayes, ensemble

df = pd.read_csv('../data/decimated_training_set_10_pct.csv')

#taking the base 10 log, gives the feature price_usd monotonic utility with respect to the target vairiables
df['price_usd']= df['price_usd'].replace(0,df['price_usd'].max(),regex=True)
df['log10_price_usd'] = df['price_usd'].apply(math.log10)


"""Handling Missing Values"""
df['prop_review_score'].fillna(df['prop_review_score'].min(), inplace = True)
df['prop_location_score2'].fillna(df['prop_location_score2'].min(), inplace = True)
df['srch_query_affinity_score'].fillna(df['srch_query_affinity_score'].min(), inplace = True)

for i in range(1,9):
	df['comp'+str(i)+'_rate'].fillna(0, inplace = True)
	df['comp'+str(i)+'_inv'].fillna(0, inplace = True)

#Note sure whether to do this now
df['visitor_hist_starrating'].fillna(0, inplace = True)
df['visitor_hist_adr_usd'].fillna(0, inplace = True)



"""Feature extraction"""
df['starrating_diff'] = abs(df['visitor_hist_starrating'] - df['prop_starrating'] )
df['usd_diff'] = abs(df['visitor_hist_adr_usd'] - df['price_usd'] ) #use log price or normal price?

#Estimate probabilitie of hotel being booked or clicked
grouped = df.groupby('prop_id', as_index=False)
df2 = grouped['booking_bool'].agg({'sum_book_bool' : np.sum, 'count':len})
df3 = grouped['click_bool'].agg({'sum_click_bool': np.sum})
result = pd.merge(df,df2, how='inner', on='prop_id')
df= pd.merge(result,df3, how='inner', on='prop_id')

df['prob_book_prop_id'] = df['sum_book_bool']/df['count']
df['prob_click_prop_id'] = df['sum_click_bool']/df['count']

###Create new variable prop_starrating_monotonic
grouped = df.groupby('booking_bool', as_index=False)
df2 =  grouped['prop_starrating'].agg({'mean_propstarrating_booked' : np.mean})
df['prop_starrating_monotonic'] =  df['prop_starrating'] - df2.mean_propstarrating_booked



""" Normalization with respect to srch_id"""
###normalize prices by mean and standard diviation###
grouped = df.groupby('srch_id', as_index=False)
df1 = grouped['log10_price_usd'].agg({'mean_price_search_id' : np.mean, 'std_price_search_id' : np.std})
result = pd.merge(df,df1, how='inner', on='srch_id')
df = result
df['price_usd_Gauss_normalzed_search_id'] = (df.log10_price_usd - df.mean_price_search_id)/df.std_price_search_id

###normalize by the maximal and minimal observation###
grouped = df.groupby('srch_id', as_index=False)
df1 = grouped['log10_price_usd'].agg({'max_price_srch_id' : np.amax, 'min_price_srch_id' : np.amin})
result = pd.merge(df,df1, how='inner', on='srch_id')
df = result
df['price_usd_normalized_search_id'] = (df.log10_price_usd-df.min_price_srch_id)/df.max_price_srch_id


""" Normalization with respect to prop_id"""
###normalize prices by mean and standard diviation###
grouped = df.groupby('prop_id', as_index=False)
df1 = grouped['log10_price_usd'].agg({'mean_price_prop_id' : np.mean, 'std_price_prop_id' : np.std})
result = pd.merge(df,df1, how='inner', on='prop_id')
df = result
df['price_usd_Gauss_normalzed_prop_id'] = (df.log10_price_usd - df.mean_price_prop_id)/df.std_price_prop_id
df['price_usd_Gauss_normalzed_prop_id'] = df['price_usd_Gauss_normalzed_prop_id'].replace(float('inf'),0,regex=True)


###normalize by the maximal and minimal observation###
grouped = df.groupby('prop_id', as_index=False)
df1 = grouped['log10_price_usd'].agg({'max_price_prop_id' : np.amax, 'min_price_prop_id' : np.amin})
result = pd.merge(df,df1, how='inner', on='prop_id')
df = result
df['price_usd_normalized_prop_id'] = (df.log10_price_usd - df.min_price_prop_id)/df.max_price_prop_id



"""Normalization with respect to srch_destination_id"""
###normalize prices by mean and standard diviation###
grouped = df.groupby('srch_destination_id', as_index=False)
df1 = grouped['log10_price_usd'].agg({'mean_price_srch_destination_id' : np.mean, 'std_price_srch_destination_id' : np.std})
result = pd.merge(df,df1, how='inner', on='srch_destination_id')
df = result
df['price_usd_Gauss_normalzed_srch_destination_id'] = (df.log10_price_usd - df.mean_price_srch_destination_id)/df.std_price_srch_destination_id


###normalize by the maximal and minimal observation###
grouped = df.groupby('srch_destination_id', as_index=False)
df1 = grouped['log10_price_usd'].agg({'max_price_srch_destination_id' : np.amax, 'min_price_srch_destination_id' : np.amin})
result = pd.merge(df,df1, how='inner', on='srch_destination_id')
df = result
df['price_usd_normalized_srch_destination_id'] = (df.log10_price_usd - df.min_price_srch_destination_id)/df.max_price_srch_destination_id

###remove infinity values and NaNs###
df['orig_destination_distance']= df['orig_destination_distance'].replace(float('NaN'),df['orig_destination_distance'].max(),regex=True)
df = df.replace(float('inf'),0,regex=True)
df= df.replace(float('NaN'),0,regex=True)


### Training###
features_engineered = ['log10_price_usd', 'starrating_diff', 'usd_diff', 'prob_book_prop_id', 'prob_click_prop_id','prop_starrating_monotonic', 'price_usd_Gauss_normalzed_search_id', 'price_usd_Gauss_normalzed_prop_id', 'price_usd_Gauss_normalzed_prop_id','price_usd_normalized_srch_destination_id', 'price_usd_normalized_prop_id', 'price_usd_normalized_search_id']

numerical_attributes = ['prop_review_score', 'prop_brand_bool', 'prop_location_score1', 'prop_location_score2', 'prop_log_historical_price', 'promotion_flag', 'srch_length_of_stay', 'srch_booking_window', 'srch_adults_count', 'srch_children_count', 'srch_room_count', 'srch_saturday_night_bool', 'srch_query_affinity_score', 'orig_destination_distance', 'random_bool']

train_set = df[numerical_attributes + features_engineered + ['booking_bool', 'click_bool']]

#Click training
y = train_set['click_bool']
X = train_set[numerical_attributes + features_engineered]
clf_click = svm.SVC(class_weight = 'auto')
clf_click.fit(X,y)

print('done with training on clicks')

#Booking training
y = train_set['booking_bool']
X = train_set[numerical_attributes + features_engineered]
clf_book = svm.SVC(class_weight = 'auto')
clf_book.fit(X,y)

print('done with training on bookings')

df= pd.read_csv('../data/decimated_training_set_1_pct.csv')

###Testing###
test_set = df[numerical_attributes + features_engineered + ['booking_bool', 'click_bool']]

X = test_set[numerical_attributes + features_engineered]
click_predictions = clf_click.predict(X)
booking_predictions = clf_book.predict(X)

print('done testing')

ref_df = df[['srch_id', 'booking_bool', 'click_bool']]
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
        idcg += (2**val.real_score-1)/np.log2(i+1)
    predict_sorted = group.sort('predict_score', ascending=False)
    dcg = 0
    for (i, (index,val)) in enumerate(predict_sorted.iterrows(), start=1):
        dcg += (2**val.real_score-1)/np.log2(i+1)
    ndcgs.append(dcg/idcg)

print np.mean(ndcgs)


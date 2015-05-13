import pandas
import numpy
from sklearn import linear_model, cross_validation, svm, tree, naive_bayes, ensemble

full_df = pandas.read_csv('../data/decimated_training_set_10_pct.csv')

numerical_attributes = ['visitor_hist_starrating', 'visitor_hist_adr_usd', 'prop_starrating', 'prop_review_score', 'prop_brand_bool', 'prop_location_score1', 'prop_location_score2', 'prop_log_historical_price', 'price_usd', 'promotion_flag', 'srch_length_of_stay', 'srch_booking_window', 'srch_adults_count', 'srch_children_count', 'srch_room_count', 'srch_saturday_night_bool', 'srch_query_affinity_score', 'orig_destination_distance', 'random_bool']

df = full_df[numerical_attributes + ['booking_bool', 'click_bool']]

df = df.fillna(df.mean())


clickers = df[df.click_bool == 1]
non_clickers = df[df.click_bool == 0][:len(clickers)]
new_frame = pandas.concat([clickers,non_clickers] )
X = new_frame[numerical_attributes]
y = new_frame['click_bool']
clicking_classifier = ensemble.GradientBoostingClassifier()
clicking_classifier.fit(X,y)



bookers = df[df.booking_bool == 1]
non_bookers = df[df.booking_bool == 0][:len(bookers)]
new_frame = pandas.concat([bookers,non_bookers] )
X = new_frame[numerical_attributes]
y = new_frame['booking_bool']
booking_classifier = ensemble.GradientBoostingClassifier()
booking_classifier.fit(X,y)



full_df = pandas.read_csv('../data/decimated_training_set_1_pct.csv')
test_set = full_df[numerical_attributes]
test_set = test_set.fillna(test_set.mean())
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

print numpy.mean(ndcgs)

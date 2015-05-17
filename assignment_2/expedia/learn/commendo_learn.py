import pandas
import numpy as np
from sklearn import linear_model, cross_validation, svm, tree, naive_bayes, ensemble
import pickle


print "loading data"
full_df = pandas.read_csv('../data/decimated_training_set_9_pct.csv')

numerical_attributes = ['visitor_hist_starrating', 'visitor_hist_adr_usd', 'prop_starrating', 'prop_review_score', 'prop_brand_bool', 'prop_location_score1', 'prop_location_score2', 'prop_log_historical_price', 'price_usd', 'promotion_flag', 'srch_length_of_stay', 'srch_booking_window', 'srch_adults_count', 'srch_children_count', 'srch_room_count', 'srch_saturday_night_bool', 'srch_query_affinity_score', 'orig_destination_distance', 'random_bool']
print "done loading data"


scores = []

def gen_aggregate_vals(df, attribs):
    print "generating aggregate values"
    hotel_groups = df.groupby('prop_id', as_index=False)
    for attrib in numerical_attributes:
        attrib_hotel_group = hotel_groups[attrib]
        agg_result = attrib_hotel_group.agg({attrib+'_mean':np.mean, attrib+'_std':np.std, attrib+'_median':np.median})
        agg_result[attrib+'_std'] = agg_result[attrib+'_std'].fillna(0)
        agg_result = agg_result.fillna(agg_result.mean())
        df = pandas.merge(df, agg_result, on='prop_id')
    print "done generating aggregate values"
    return df

#agg_attribs = numerical_attributes

for attribute in numerical_attributes + [None]:
#if True:

    agg_attribs = [attribute]

    df = full_df[numerical_attributes + ['booking_bool', 'click_bool', 'prop_id']]
    if attribute:
        df = gen_aggregate_vals(df, agg_attribs)
    df = df.fillna(df.mean())
    df = df.drop('prop_id', 1)

    model = ensemble.GradientBoostingClassifier
    #model = linear_model.SGDClassifier

    print "training clicker model"
    clickers = df[df.click_bool == 1]
    non_clickers = df[df.click_bool == 0][:len(clickers)]
    new_frame = pandas.concat([clickers,non_clickers] )
    X = new_frame.drop(['click_bool', 'booking_bool'],1)
    y = new_frame['click_bool']
    clicking_classifier = model()
    clicking_classifier.fit(X,y)
    pickle.dump(clicking_classifier, open("clicking_classifier.pickle","wb"))



    print "training booking model"
    bookers = df[df.booking_bool == 1]
    non_bookers = df[df.booking_bool == 0][:len(bookers)]
    new_frame = pandas.concat([bookers,non_bookers] )
    X = new_frame.drop(['click_bool', 'booking_bool'],1)
    y = new_frame['booking_bool']
    booking_classifier = model()
    booking_classifier.fit(X,y)
    pickle.dump(booking_classifier, open("booking_classifier.pickle","wb"))


    print "generating prediction for test set"
    full_df = pandas.read_csv('../data/decimated_training_set_1_pct.csv')
    test_set = full_df[numerical_attributes+['prop_id']]
    if attribute:
        test_set = gen_aggregate_vals(test_set, agg_attribs)
    test_set = test_set.fillna(test_set.mean())
    test_set = test_set.drop('prop_id', 1)
    click_predictions = clicking_classifier.predict(test_set)
    booking_predictions = booking_classifier.predict(test_set)


    ref_df = full_df.loc[:,('srch_id', 'booking_bool', 'click_bool', 'prop_id')]
    ref_df['real_score'] = (ref_df.booking_bool * 5  + ref_df.click_bool)
    ref_df['predict_booking'] = booking_predictions
    ref_df['predict_click'] = click_predictions
    ref_df['predict_score'] = ref_df.predict_booking * 5 + ref_df.predict_click
    grouped = ref_df.groupby('srch_id')

    print attribute
    print "generating NDCG score"
    ndcgs = []
    prediction_file = open("prediction.csv", "w")
    prediction_file.write("SearchId,PropertyId\n")
    for name, group in grouped:
        real_sorted = group.sort('real_score', ascending=False)
        idcg = 0
        for (i, (index,val)) in enumerate(real_sorted.iterrows(), start=1):
            idcg += (2**val.real_score-1)/np.log2(i+1)
        predict_sorted = group.sort('predict_score', ascending=False)
        dcg = 0
        for (i, (index,val)) in enumerate(predict_sorted.iterrows(), start=1):
            prediction_file.write(str(name)+","+str(val.prop_id)+"\n")
            dcg += (2**val.real_score-1)/np.log2(i+1)
        ndcgs.append(dcg/idcg)
    prediction_file.close()
    ndcg_mean = np.mean(ndcgs)
    print ndcg_mean
    print
    scores.append((ndcg_mean, attribute))

print scores

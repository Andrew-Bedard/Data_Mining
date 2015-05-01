import pandas
from sklearn import linear_model, cross_validation, svm, tree, naive_bayes, ensemble

df = pandas.read_csv('../data/decimated_training_set.csv')
df = df[['srch_id', 'site_id', 'visitor_location_country_id', 'prop_country_id', 'prop_id', 'prop_starrating', 'price_usd', 'site_id', 'srch_adults_count', 'srch_children_count', 'click_bool']].dropna()[:5000]
X = df[['srch_id', 'site_id', 'visitor_location_country_id', 'prop_country_id', 'prop_id', 'prop_starrating', 'price_usd', 'site_id', 'srch_adults_count', 'srch_children_count']]
y = df['click_bool']

classifier = svm.SVC()


scores = cross_validation.cross_val_score(classifier, X, y, cv=5)
print scores

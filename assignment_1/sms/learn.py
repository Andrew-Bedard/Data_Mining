from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import cross_validation, naive_bayes
from nltk.corpus import stopwords
import pandas
import string


def tokenize_sms(sms):
    sms = filter(lambda x: x in string.printable, sms)
    words = word_tokenize(sms)
    stemmer = SnowballStemmer("english")
    stemmed_words = map(stemmer.stem, words)
    without_stop_words = [word for word in stemmed_words if not word in stopwords.words("english")]
    return " ".join(set(without_stop_words))

df = pandas.read_csv("clean_sms_collection.csv", sep=";", escapechar="\\")
df['tokenized_message'] = df["text"].apply(tokenize_sms)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['tokenized_message'])
y = df['label']


classifier = naive_bayes.GaussianNB()
scores = cross_validation.cross_val_score(classifier, X.toarray(), y, cv=10)
mean_score = scores.mean()
print "10-fold cross-validation score", mean_score

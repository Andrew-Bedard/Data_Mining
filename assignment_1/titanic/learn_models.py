import pandas
from sklearn import linear_model, cross_validation, svm, tree, naive_bayes, ensemble
df = pandas.read_csv('train.csv')




def train_on_model(classifier, X, y, classifier_name):
    scores = cross_validation.cross_val_score(classifier, X, y, cv=cross_validation.LeaveOneOut(len(y)))
    print classifier_name
    mean_score = scores.mean()
    print "leave one out score", mean_score
    return mean_score


def train_on_different_models(X, y, training_fields):
    print training_fields
    scores = [training_fields]
    scores.append(train_on_model(linear_model.LogisticRegression(), X, y, "logistic regression"))
    scores.append(train_on_model(svm.SVC(), X, y, "support vector machine"))
    scores.append(train_on_model(tree.DecisionTreeClassifier(), X, y, "decision tree"))
    scores.append(train_on_model(naive_bayes.GaussianNB(), X, y, "naive bayes"))
    scores.append(train_on_model(ensemble.RandomForestClassifier(), X, y, "random forest"))
    print 
    return scores

print "always guessing non survival"
y = df['Survived']
print 1-y.mean()
print 

X = df[['Pclass', 'Sex']]
X['Sex'] = X['Sex'].apply(lambda x: 0 if x=='male' else 1)
class_sex_scores = train_on_different_models(X,y, "class and sex")

X['Family_size'] = df.Parch + df.SibSp
class_sex_fam_scores = train_on_different_models(X,y, "class, sex and family size")

X['Known_cabin'] = df.Cabin.apply(lambda x: 1 if pandas.notnull(x) else 0)
class_sex_fam_cabin_scores = train_on_different_models(X,y, "class, sex, family size and known cabin")

X['Fare'] = df['Fare']
class_sex_fam_cabin_fare_scores = train_on_different_models(X,y, "class, sex, family size, known cabin and fare")

X['Age'] = df['Age'].fillna(df['Age'].mean())
class_sex_fam_cabin_fare_age_scores = train_on_different_models(X,y, "class, sex, family size, known cabin, fare and age")


def find_port_starboard(cabin):
    if " " in cabin:
        cabin = cabin.split()[0]
    if len(cabin) == 1:
        return 0.5
    if cabin[0] == 'E':
        return 0.5
    return int(cabin[1:])%2

X['Port_star'] = df['Cabin'].fillna('X').apply(find_port_starboard)
class_sex_fam_cabin_fare_age_portstar_scores = train_on_different_models(X,y, "class, sex, family size, known cabin, fare, age and ship side")

X = X[['Pclass', 'Sex', 'Known_cabin', 'Fare']]
X['Parch'] = df['Parch']
X['SibSp'] = df['SibSp']
class_sex_sibspouse_parentchild_cabin_fare_scores = train_on_different_models(X,y, "class, sex, sibling/spouse, parents/children, known cabin and fare")

X = X[['Pclass', 'Sex', 'Known_cabin']]
class_sex_cabin_scores = train_on_different_models(X,y, "class, sex and known cabin")

print "fields used & logistic regression & support vector machine & decision tree & naive bayes & random forest\\\\"
print " & ".join(map(str,class_sex_scores)) + " \\\\"
print " & ".join(map(str,class_sex_fam_scores)) + " \\\\"
print " & ".join(map(str,class_sex_cabin_scores)) + " \\\\"
print " & ".join(map(str,class_sex_fam_cabin_scores)) + " \\\\"
print " & ".join(map(str,class_sex_fam_cabin_fare_scores)) + " \\\\"
print " & ".join(map(str,class_sex_sibspouse_parentchild_cabin_fare_scores)) + " \\\\"
print " & ".join(map(str,class_sex_fam_cabin_fare_age_scores)) + " \\\\"
print " & ".join(map(str,class_sex_fam_cabin_fare_age_portstar_scores)) + " \\\\"


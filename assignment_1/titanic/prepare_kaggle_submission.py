import pandas
from sklearn import linear_model, cross_validation, svm, tree, naive_bayes

df = pandas.read_csv('train.csv')
X = df[['Pclass', 'Sex']]
X['Sex'] = X['Sex'].apply(lambda x: 0 if x=='male' else 1)
X['Family_size'] = df.Parch + df.SibSp
X['Known_cabin'] = df.Cabin.apply(lambda x: 1 if pandas.notnull(x) else 0)
X['Fare'] = df['Fare']
y = df['Survived']

classifier = tree.DecisionTreeClassifier()
classifier.fit(X,y)

df = pandas.read_csv('test.csv')
X = df[['Pclass', 'Sex']]
X['Sex'] = X['Sex'].apply(lambda x: 0 if x=='male' else 1)
X['Family_size'] = df.Parch + df.SibSp
X['Known_cabin'] = df.Cabin.apply(lambda x: 1 if pandas.notnull(x) else 0)
X['Fare'] = df['Fare']
X['Fare'] = df['Fare'].fillna(df['Fare'].mean())

predictions = classifier.predict(X)

f = open("submission.csv", "w")
f.write("PassengerId,Survived\n")
for (passengerId, prediction) in zip(df.PassengerId, predictions):
    f.write(str(passengerId)+","+str(prediction)+"\n")
f.close()


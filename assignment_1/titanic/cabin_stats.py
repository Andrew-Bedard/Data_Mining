import pandas
import matplotlib as plt
from matplotlib import pyplot


df = pandas.read_csv('train.csv')

def find_first_cabin_number(cabin):
    if " " in cabin:
        cabin = cabin.split()[0]
    if len(cabin) == 1:
        return 0
    return int(cabin[1:])%2


def find_first_cabin_deck(cabin):
    deck = cabin[0]
    if deck == 'T':
        return 0
    return ord(deck)-ord('A')

df_with_cabins = df[pandas.notnull(df['Cabin'])]
df_without_deck_E = df_with_cabins[~df_with_cabins.Cabin.str.contains('E')]

print df_without_deck_E['Cabin'].apply(find_first_cabin_number).corr(df_without_deck_E['Survived'])


print df_with_cabins['Cabin'].apply(find_first_cabin_deck).corr(df_with_cabins['Survived'])

df_with_cabins['Deck'] = df_with_cabins['Cabin'].apply(lambda x: x[0])

temp = df_with_cabins.groupby('Deck').Survived.sum()/df_with_cabins.groupby('Deck').Survived.count()
print temp


fig = pyplot.figure()
ax = fig.add_subplot(111)
temp.plot(kind = 'bar')
pyplot.xlabel("Deck")
pyplot.ylabel("Probability of Survival")
pyplot.savefig('../report/survival_per_deck.png')

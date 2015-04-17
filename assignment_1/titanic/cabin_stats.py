import pandas

df = pandas.read_csv('train.csv')

def find_first_cabin_number(cabin):
    if " " in cabin:
        cabin = cabin.split()[0]
    if len(cabin) == 1:
        return 0
    return int(cabin[1:])%2


df_with_cabins = df[pandas.notnull(df['Cabin'])]
df_without_deck_E = df_with_cabins[~df_with_cabins.Cabin.str.contains('E')]

print df_without_deck_E['Cabin'].apply(find_first_cabin_number).corr(df_without_deck_E['Survived'])

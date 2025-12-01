import pandas as pd

df = pd.read_csv('data/student-mat.csv', sep=';')
df_encoded = pd.get_dummies(df, drop_first=True)
X = df_encoded.drop('G3', axis=1)

print('Total fitur:', len(X.columns))
print('\nNama fitur:')
for i, col in enumerate(X.columns, 1):
    print(f'{i}. {col}')

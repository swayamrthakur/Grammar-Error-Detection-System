import pandas as pd

df = pd.read_csv('Data/final_dataset.csv')

print('Total rows:', len(df))
print('\nLabel distribution:')
print(df['label'].value_counts())

print('\nSample sentences label 0 (Incorrect):')
for s in df[df['label']==0]['sentence'].head(5).tolist():
    print(' -', s)

print('\nSample sentences label 1 (Correct):')
for s in df[df['label']==1]['sentence'].head(5).tolist():
    print(' -', s)
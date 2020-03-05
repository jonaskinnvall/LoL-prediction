import pandas as pd

# Read csv file into pandas data frame
df = pd.read_csv('data/games.csv')

# Print shape of df
print('Data Frame shape: ', df.shape)

# Print columns of df
print('Columns of data: ', df.columns)

# Print first 5 rows of df
print(df.head(5))

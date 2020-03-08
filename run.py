# Lib imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Module imports
from train import trainNN
from test import testNN

df = pd.read_csv('data/games.csv')

# Drop duplicates
df = df.drop_duplicates()

# Remove games that lasted shorter than 10 minutes
df = df[~((df['gameDuration'] / 60) <= 10)]

y = df['winner']
X = df[['firstTower', 'firstInhibitor', 'firstDragon',
        't1_towerKills', 't1_inhibitorKills', 't1_dragonKills',
        't1_baronKills',
        't2_towerKills', 't2_inhibitorKills', 't2_dragonKills',
        't2_baronKills']]

# Change pandas data frames to numpy ndarrays
X = X.values
y = y.values

# Scale the data and change valeus indicating
# winners from 1 and 2 to 0 and 1 respectively
X = MinMaxScaler().fit_transform(X)
y = np.where(y == 1, 0, y)
y = np.where(y == 2, 1, y)

# Split training set into training set and validation set
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    shuffle=False,
                                                    random_state=42)

# Let user choose if they want to train ANN or test it
print("Do you want to train or test the ANN?")
response = None
while response not in {"train", "test"}:
    response = input("Please enter 'train' or 'test': ")

if response == 'train':
    trainNN(X_train, y_train)
elif response == 'test':
    testNN(X_test, y_test)

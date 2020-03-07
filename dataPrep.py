import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

""" pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000) """
sns.set()

# Read csv file into pandas data frame
df = pd.read_csv('data/games.csv')

# ----------- CONSOLE PRINTING -------------

# Print shape of df
print('Data Frame shape: ', df.shape, '\n')

# Print first 5 rows of df
print('First 5 elements of data: \n', df.head(5), '\n')

# Use info and describe to print a concise summary
# and some descriptive statistics of the data frame
print('Data info: \n', df.info(), '\n')
print('Describe: \n', df.describe(), '\n')

df = df.drop(['gameId', 'creationTime', 'seasonId'], axis=1)

# See if there are any duplicated rows
print('Duplicates: \n', df.duplicated().sum(), '\n')

# Display how balanced the dataset is w.r.t winner
print('Winner distribution: \n', df.groupby('winner').size(), '\n')

# See average amount of different kills for the winners
stats_avg = df[['winner', 't1_towerKills', 't1_inhibitorKills',
                't1_baronKills', 't1_dragonKills', 't1_riftHeraldKills',
                't2_towerKills', 't2_inhibitorKills', 't2_baronKills',
                't2_dragonKills', 't2_riftHeraldKills']]

print('Avg kills: \n', stats_avg.groupby('winner').mean(), '\n')

# ---------- FIGURE PLOTTING -----------

plt.figure(1, figsize=(10, 6))
plt.subplot(1, 2, 1)
sns.distplot(df[['gameDuration']] / 60)

df = df[~((df['gameDuration'] / 60) <= 10)]
plt.subplot(1, 2, 2)
sns.distplot(df[['gameDuration']] / 60)

# Drop gameDuration
df = df.drop(['gameDuration'], axis=1)

# See correlation between columns in df,
# specifically looking for which statistic
# that is most important to win the match
firsts = df[['winner', 'firstBlood', 'firstTower', 'firstInhibitor',
             'firstBaron', 'firstDragon', 'firstRiftHerald']]

t1kills = df[['winner', 't1_towerKills', 't1_inhibitorKills',
              't1_baronKills', 't1_dragonKills', 't1_riftHeraldKills']]


t2kills = df[['winner', 't2_towerKills', 't2_inhibitorKills',
              't2_baronKills', 't2_dragonKills', 't2_riftHeraldKills']]

t1champs = df[['winner', 't1_champ1id', 't1_champ2id', 't1_champ3id',
               't1_champ4id', 't1_champ5id']]

t2champs = df[['winner', 't2_champ1id', 't2_champ2id', 't2_champ3id',
               't2_champ4id', 't2_champ5id']]

t1bans = df[['winner', 't1_ban1', 't1_ban2', 't1_ban3',
             't1_ban4', 't1_ban5']]

t2bans = df[['winner', 't2_ban1', 't2_ban2', 't2_ban3',
             't2_ban4', 't2_ban5']]

# KILLS CORRELATIONS
plt.figure(2, figsize=(10, 7))
sns.heatmap(firsts.corr(), annot=True)

plt.figure(3, figsize=(10, 7))
sns.heatmap(t1kills.corr(), annot=True)
plt.figure(4, figsize=(10, 7))
sns.heatmap(t2kills.corr(), annot=True)

# CHAMPS AND BANS CORRELATIONS
plt.figure(5, figsize=(10, 7))
sns.heatmap(t1champs.corr(), annot=True)
plt.figure(6, figsize=(10, 7))
sns.heatmap(t2champs.corr(), annot=True)

plt.figure(7, figsize=(10, 7))
sns.heatmap(t1bans.corr(), annot=True)
plt.figure(8, figsize=(10, 7))
sns.heatmap(t2bans.corr(), annot=True)

# -------- DATA SELECTION ---------

# Will be done in training and testing files separately,
# just for testing around here

y = df['winner']
X = df[['firstTower', 'firstInhibitor', 'firstDragon',
        't1_towerKills', 't1_inhibitorKills', 't1_dragonKills',
        't1_baronKills',
        't2_towerKills', 't2_inhibitorKills', 't2_dragonKills',
        't2_baronKills']]

plt.figure(9, figsize=(13, 7))
sns.boxplot(data=X)

print('X shape: ', X.shape)
print('y shape: ', y.shape)

plt.show()

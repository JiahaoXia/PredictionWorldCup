__author__ = 'XJH'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.ticker as plticker
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# load data
world_cup = pd.read_csv('H:/Machine Learning/results prediction of World Cup/World Cup 2018 Dataset.csv')
results = pd.read_csv('H:/Machine Learning/results prediction of World Cup/results.csv')

# print(world_cup.head())
# print(results.head())

# goal difference and winner
winner = []
for i in range (len(results['home_team'])):
	if results['home_score'][i] > results['away_score'][i]:
		winner.append(results['home_team'][i])
	elif results['home_score'][i] < results['away_score'][i]:
		winner.append(results['away_team'][i])
	else:
		winner.append('Draw')
results['winning_team'] = winner

results['goal_differences'] = np.absolute(results['home_score'] - results['away_score'])
# print(results.head())

df = results[(results['home_team'] == 'Nigeria') | (results['away_team'] == 'Nigeria')]
nigeria = df.iloc[:]
# print(nigeria.head())

# create a column for year
# year = []
# for row in nigeria['date']:
# 	year.append(int(row[:4]))
# nigeria['match_year'] = year
# nigeria_1930 = nigeria[nigeria.match_year >= 1930]
# print(nigeria_1930.count())

worldcup_teams = world_cup['Team']
# print(worldcup_teams.head())

df_teams_home = results[results['home_team'].isin(worldcup_teams)]
df_teams_away = results[results['away_team'].isin(worldcup_teams)]
# print(df_teams_away.count())
df_teams = pd.concat((df_teams_home, df_teams_away))
# print(df_teams.count())

# create a year column to drop games before 1930
year = []
for row in df_teams['date']:
	year.append(int(row[:4]))
df_teams['match_year'] = year
df_teams_1930 = df_teams[df_teams.match_year >= 1930]
# print(df_teams_1930.head())

# dropping columns that will not affect prediction
df_teams_1930_copy = df_teams_1930
df_teams_1930 = df_teams_1930_copy.drop(['date', 'home_score', 'away_score',
									'tournament', 'city', 'country', 'neutral',
									'goal_differences', 'match_year'], axis=1)
print(df_teams_1930.head())

# building the model
# prediction label 2 = home team has won
# 1 = Draw
# 0 = away team has won
df_teams_1930 = df_teams_1930.reset_index(drop=True)
df_teams_1930.loc[df_teams_1930.winning_team == df_teams_1930.home_team, 'winning_team'] = 2
df_teams_1930.loc[df_teams_1930.winning_team == 'Draw', 'winning_team'] = 1
df_teams_1930.loc[df_teams_1930.winning_team == df_teams_1930.away_team, 'winning_team'] = 0
print(df_teams_1930.head())

# get dummy variables
final = pd.get_dummies(df_teams_1930, prefix=['home_team', 'away_team'],
					   columns=['home_team', 'away_team'])
# print(final.head())

# separate X and y
X = final.drop(['winning_team'], axis=1)
y = final['winning_team']
y = y.astype('int')

# separate train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
train_score = logreg.score(X_train, y_train)
test_score = logreg.score(X_test, y_test)

print('Training set accuracy: %.3f' % train_score)
print('Testing set accuracy: %.3f' % test_score)

# load FIFA ranks
# fifa_ranks = pd.read_csv('H:/Machine Learning/results prediction of World Cup/fifa_ranking.csv')
# year = []
# for row in fifa_ranks['rank_date']:
# 	year.append(10*int(row[:4]) + int(row[6]))
# fifa_ranks['year'] = year
# fifa_ranks_20186 = fifa_ranks[fifa_ranks.year > 20185]
# fifa_ranks_20186 = fifa_ranks_20186.drop(['country_abrv', 'total_points', 'previous_points',
# 										  'rank_change', 'cur_year_avg', 'cur_year_avg_weighted',
# 										  'last_year_avg', 'last_year_avg_weighted', 'two_year_ago_avg',
# 										  'two_year_ago_weighted', 'three_year_ago_avg', 'three_year_ago_weighted',
# 										  'confederation', 'rank_date', 'year'], axis=1)
# print(fifa_ranks_20186.head())
#
# worldcup_schedule = pd.read_csv('H:/Machine Learning/results prediction of World Cup/fifa-world-cup-2018-RussianStandardTime.csv')
# round_number = worldcup_schedule['Round Number']
ranking = pd.read_csv('H:/Machine Learning/results prediction of World Cup/fifa_rankings.csv')
fixtures = pd.read_csv('H:/Machine Learning/results prediction of World Cup/fixtures.csv')

# list for storing the group stage games
pred_set = []
fixtures.insert(1, 'first_position', fixtures['Home Team'].map(ranking.set_index('Team')['Position']))
fixtures.insert(2, 'second_position', fixtures['Away Team'].map(ranking.set_index('Team')['Position']))
# print(fixtures.head())
fixtures = fixtures.iloc[:48, :]
# print(fixtures.tail())
for index, row in fixtures.iterrows():
	if row['first_position'] < row['second_position']:
		pred_set.append({'home_team' : row['Home Team'], 'away_team' : row['Away Team'], 'winning_team' : None})
	else:
		pred_set.append({'home_team' : row['Away Team'], 'away_team' : row['Home Team'], 'winning_team' : None})
pred_set = pd.DataFrame(pred_set)
backup_pred_set = pred_set
print(pred_set.head())

pred_set = pd.get_dummies(pred_set, prefix=['home_team', 'away_team'], columns=['home_team', 'away_team'])
# add missing columns compared to the model's training dataset
missing_cols = set(final.columns)- set(pred_set.columns)
for c in missing_cols:
	pred_set[c] = 0
pred_set = pred_set[final.columns]
# remove winning team column
pred_set = pred_set.drop(['winning_team'], axis=1)
print(pred_set.head())

# group matches
predictions = logreg.predict(pred_set)
for i in range(fixtures.shape[0]):
	print(backup_pred_set.iloc[i, 1] + " and " + backup_pred_set.iloc[i, 0])
	if predictions[i] == 2:
		print("Winner: " + backup_pred_set.iloc[i, 1])
	elif predictions[i] == 1:
		print("Draw")
	elif predictions[i] == 0:
		print("Winner: " + backup_pred_set.iloc[i, 0])
	print('Probability of ' + backup_pred_set.iloc[i, 1] + 'winning: %.3f' % (logreg.predict_proba(pred_set)[i][2]))
	print('Probability of Draw: %.3f' % (logreg.predict_proba(pred_set)[i][1]))
	print('Probability of ' + backup_pred_set.iloc[i, 0] + 'winning: %.3f' % (logreg.predict_proba(pred_set)[i][0]))

print('***done***')
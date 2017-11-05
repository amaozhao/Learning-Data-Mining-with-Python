# # coding: utf-8

import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier


data_filename = 'leagues_NBA_2016_games.csv'
results = pd.read_csv(data_filename)
results["HomeWin"] = results["PTS"] < results["PTS.1"]
results["HomeLastWin"] = False
results["VisitorLastWin"] = False
y_true = results["HomeWin"].values

won_last = defaultdict(int)

for index, row in results.iterrows():  # Note that this is not efficient
    home_team = row["Home/Neutral"]
    visitor_team = row["Visitor/Neutral"]
    row["HomeLastWin"] = won_last[home_team]
    row["VisitorLastWin"] = won_last[visitor_team]
    results.ix[index] = row
    # Set current win
    won_last[home_team] = row["HomeWin"]
    won_last[visitor_team] = not row["HomeWin"]
results.ix[20:25]


clf = DecisionTreeClassifier(random_state=14)
X_previouswins = results[["HomeLastWin", "VisitorLastWin"]].values
clf = DecisionTreeClassifier(random_state=14)
scores = cross_val_score(clf, X_previouswins, y_true, scoring='accuracy')
print("Using just the last result from the home and visitor teams")
print("Accuracy: {0:.1f}%".format(np.mean(scores) * 100))

# What about win streaks?
results["HomeWinStreak"] = 0
results["VisitorWinStreak"] = 0
# Did the home and visitor teams win their last game?
win_streak = defaultdict(int)

for index, row in results.iterrows():  # Note that this is not efficient
    home_team = row["Home/Neutral"]
    visitor_team = row["Visitor/Neutral"]
    row["HomeWinStreak"] = win_streak[home_team]
    row["VisitorWinStreak"] = win_streak[visitor_team]
    results.ix[index] = row
    # Set current win
    if row["HomeWin"]:
        win_streak[home_team] += 1
        win_streak[visitor_team] = 0
    else:
        win_streak[home_team] = 0
        win_streak[visitor_team] += 1

clf = DecisionTreeClassifier(random_state=14)
X_winstreak = results[
    [
        "HomeLastWin", "VisitorLastWin",
        "HomeWinStreak", "VisitorWinStreak"
    ]
].values
scores = cross_val_score(clf, X_winstreak, y_true, scoring='accuracy')
print("Using whether the home team is ranked higher")
print("Accuracy: {0:.1f}%".format(np.mean(scores) * 100))


ladder = pd.read_csv(
    'leagues_NBA_2014_standings_expanded-standings.csv', skiprows=[0])

# We can create a new feature -- HomeTeamRanksHigher\
results["HomeTeamRanksHigher"] = 0
for index, row in results.iterrows():
    home_team = row["Home/Neutral"]
    visitor_team = row["Visitor/Neutral"]
    if home_team == "New Orleans Hornets":
        home_team = "Charlotte Bobcats"
    elif visitor_team == "New Orleans Hornets":
        visitor_team = "New Orleans Hornets"
    home_rank = ladder[ladder["Team"] == home_team]["Rk"].values[0]
    visitor_rank = ladder[ladder["Team"] == visitor_team]["Rk"].values[0]
    row["HomeTeamRanksHigher"] = int(home_rank > visitor_rank)
    results.ix[index] = row
results[:5]

X_homehigher = results[
    ["HomeLastWin", "VisitorLastWin", "HomeTeamRanksHigher"]].values
clf = DecisionTreeClassifier(random_state=14)
scores = cross_val_score(clf, X_homehigher, y_true, scoring='accuracy')
print("Using whether the home team is ranked higher")
print("Accuracy: {0:.1f}%".format(np.mean(scores) * 100))


parameter_space = {
    "max_depth": [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        11, 12, 13, 14, 15, 16, 17, 18, 19, 20
    ],
}
clf = DecisionTreeClassifier(random_state=14)
grid = GridSearchCV(clf, parameter_space)
grid.fit(X_homehigher, y_true)
print("Accuracy: {0:.1f}%".format(grid.best_score_ * 100))

# Who won the last match? We ignore home/visitor for this bit
last_match_winner = defaultdict(int)
results["HomeTeamWonLast"] = 0

for index, row in results.iterrows():
    home_team = row["Home/Neutral"]
    visitor_team = row["Visitor/Neutral"]
    # Sort for a consistent ordering
    teams = tuple(sorted([home_team, visitor_team]))
    # Set in the row, who won the last encounter
    row["HomeTeamWonLast"] = 1 if last_match_winner[
        teams] == row["Home/Neutral"] else 0
    results.ix[index] = row
    # Who won this one?
    winner = row["Home/Neutral"] if row["HomeWin"] else row["Visitor/Neutral"]
    last_match_winner[teams] = winner
results.ix[:5]


X_home_higher = results[
    ["HomeTeamRanksHigher", "HomeTeamWonLast"]].values
clf = DecisionTreeClassifier(random_state=14)
scores = cross_val_score(clf, X_home_higher, y_true, scoring='accuracy')
print("Using whether the home team is ranked higher")
print("Accuracy: {0:.1f}%".format(np.mean(scores) * 100))

encoding = LabelEncoder()
encoding.fit(results["Home/Neutral"].values)
home_teams = encoding.transform(results["Home/Neutral"].values)
visitor_teams = encoding.transform(results["Visitor/Neutral"].values)
X_teams = np.vstack([home_teams, visitor_teams]).T

onehot = OneHotEncoder()
X_teams = onehot.fit_transform(X_teams).todense()

clf = DecisionTreeClassifier(random_state=14)
scores = cross_val_score(clf, X_teams, y_true, scoring='accuracy')
print("Accuracy: {0:.1f}%".format(np.mean(scores) * 100))

# 随机森林
clf = RandomForestClassifier(random_state=14)
scores = cross_val_score(clf, X_teams, y_true, scoring='accuracy')
print("Using full team labels is ranked higher")
print("Accuracy: {0:.1f}%".format(np.mean(scores) * 100))

X_all = np.hstack([X_home_higher, X_teams])
print(X_all.shape)

clf = RandomForestClassifier(random_state=14)
scores = cross_val_score(clf, X_all, y_true, scoring='accuracy')
print("Using whether the home team is ranked higher")
print("Accuracy: {0:.1f}%".format(np.mean(scores) * 100))

# n_estimators=10, criterion='gini', max_depth=None,
# min_samples_split=2, min_samples_leaf=1,
# max_features='auto',
# max_leaf_nodes=None, bootstrap=True,
# oob_score=False, n_jobs=1,
# random_state=None, verbose=0, min_density=None, compute_importances=None
parameter_space = {
    "max_features": [2, 10, 'auto'],
    "n_estimators": [100, ],
    "criterion": ["gini", "entropy"],
    "min_samples_leaf": [2, 4, 6],
}
clf = RandomForestClassifier(random_state=14)
grid = GridSearchCV(clf, parameter_space)
grid.fit(X_all, y_true)
print("Accuracy: {0:.1f}%".format(grid.best_score_ * 100))
print(grid.best_estimator_)

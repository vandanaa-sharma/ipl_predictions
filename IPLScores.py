import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

pd.options.mode.chained_assignment = None

data = pd.read_csv(r'data\IPL Matches 2008-2020.csv') 
df = pd.DataFrame(data, columns= ['team1', 'team2', 'toss_winner','toss_decision', 'winner'])

matches = pd.read_csv(r'data\ipl-2021-matches.csv') 
df_matches = pd.DataFrame(matches, columns= ['Team1', 'Team2', 'Toss Winner','Toss Decision', 'Winner'])

result = [None]*df_matches.shape[0]
accuracy = [None]*df_matches.shape[0]

for index, row in df_matches.iterrows():
    t1 = str(row['Team1'])
    t2 = str(row['Team2'])
    toss_winner = row['Toss Winner']
    if type(toss_winner) is float:
        break
    toss_decision = str(row['Toss Decision'])

    if(toss_decision == 'bat'):
        toss_decision = int(101)
    else:
        toss_decision = int(100)

    df1 = df.copy()

    if (df1.empty):
        continue

    le = LabelEncoder()
    df1['toss_winner'] = le.fit_transform(df1['toss_winner'])
    df1['winner'] = le.fit_transform(df1['winner'])
    df1['team1'] = le.fit_transform(df1['team1'])
    df1['team2'] = le.fit_transform(df1['team2'])
    
    df1.loc[(df1.toss_decision =='bat'), 'toss_decision'] = 101
    df1.loc[(df1.toss_decision =='field'), 'toss_decision'] = 100

    X = df1.drop(columns=['winner'])
    Y = df1['winner']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    
    dt = DecisionTreeClassifier()
    dt.fit(X, Y)
    test_data = le.transform([t1, t2, toss_winner])
    test_data = np.append(test_data, [toss_decision])

    print(row['Winner'], " | DT ", le.inverse_transform(dt.predict([test_data]))[0])
    result[index] = le.inverse_transform(dt.predict([test_data]))[0]

    rf = RandomForestClassifier()
    rf.fit(X, Y)
    print(row['Winner'], " | RF ", le.inverse_transform(rf.predict([test_data]))[0])
    # result[index] = le.inverse_transform(rf.predict([test_data]))[0]

    dt.fit(X_train, Y_train)
    accuracy[index] = str(round(dt.score(X_test, Y_test)*100, 2)) + "%"

    rf.fit(X_train, Y_train)
    print(str(round(rf.score(X_test, Y_test)*100, 2)) + "%")

df_matches['Predicted Winner'] = result
df_matches['Accuracy'] = accuracy

df_matches.to_csv("Results.csv")
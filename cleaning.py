import pandas as pd
import numpy as np


pd.set_option('display.max_columns',50)

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

## Drop non-useful columns

train.drop(columns = ['PassengerId','Name','Ticket','Cabin',],inplace = True)

## Replace values with 0 if female , 1 if male

train['Sex'] = train['Sex'].map({'female': 0, 'male': 1})
test['Sex'] = test['Sex'].map({'female': 0, 'male': 1})

## Fill missing values of Age with mean

median= train['Age'].median()
train['Age'].fillna(median,inplace=True)


## replace c = 1 , q = 2, s = 3

train['Embarked'] = train['Embarked'].map({'C': 1, 'Q': 2, 'S': 3})

## replace 2 missing values of Embarked with mean

mean = train['Embarked'].mean()
train['Embarked'].fillna(mean,inplace=True)

df_X = train[train.columns[train.columns != 'Survived']].copy()

df_y = train['Survived'].copy()

X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.3, random_state=25)






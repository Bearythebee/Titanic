import pandas as pd
import numpy as np


pd.set_option('display.max_columns',50)

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

columns = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

#for i in columns:
#    print(train[i].isnull().values.any())

#for i in columns:
#    if i == "Survived":
#        continue
#    else:
#        print(test[i].isnull().values.any())


##### "Sex" #####
## Replace values with 0 if female , 1 if male

train['Sex'] = train['Sex'].map({'female': 0, 'male': 1})
test['Sex'] = test['Sex'].map({'female': 0, 'male': 1})

##### "Age" #####
## Fill missing values of Age with mean

train_median= train['Age'].median()
train['Age'].fillna(train_median,inplace=True)

test_median= test['Age'].median()
test['Age'].fillna(test_median,inplace=True)

##### "Embarked" #####
## replace c = 1 , q = 2, s = 3

train['Embarked'] = train['Embarked'].map({'C': 1, 'Q': 2, 'S': 3})
test['Embarked'] = test['Embarked'].map({'C': 1, 'Q': 2, 'S': 3})

## replace 2 missing values of Embarked with mean

train_mean = train['Embarked'].mean()
train['Embarked'].fillna(train_mean,inplace=True)

test_mean = train['Embarked'].mean()
train['Embarked'].fillna(test_mean,inplace=True)


##### "Fare" #####
test_median= test['Fare'].median()
test['Fare'].fillna(test_median,inplace=True)


train.to_csv('cleaned_train.csv',index=False)
test.to_csv('cleaned_test.csv',index=False)





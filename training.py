import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


pd.set_option('display.max_columns',50)

train = pd.read_csv(r"C:\Users\Admin\Desktop\Kaggle datasets\Titanic\train.csv")


### Dropping uneccesary columns

### Fill missing data with mean

median= train['Age'].median()
train['Age'].fillna(median,inplace=True)


## Drop non-useful columns

train.drop(columns = ['PassengerId','Name','Ticket','Cabin'],inplace = True)
print(train.columns)
### Survival rate against sex

## Replace values with 0 if female , 1 if male

train['Sex'] = train['Sex'].map({'female': 0, 'male': 1})

def gender_survival_rate(train):
    try:
        female_survival_rate = train["Survived"][train["Sex"] == 0].value_counts(normalize = True)[1]*100
    except:
        female_survival_rate = 0.0
    try:
        male_survival_rate = train["Survived"][train["Sex"] == 1].value_counts(normalize = True)[1]*100
    except:
        male_survival_rate = 0.0
    print("----------")
    print("Survival rate and Gender")
    print("----------")
    print("Female survival rate = " + str(female_survival_rate)+"%")
    print("Male survival rate = " + str(male_survival_rate)+ "%")
    print("----------")

gender_survival_rate(train)

### Survival rate against ticket class(Pclass)

def ticket_class_survival_rate(train,classes):
    print("Survival rate and Ticket class")
    print("----------")
    for i in range(1,classes+1):
        try:
            Pclass = train["Survived"][train["Pclass"] == i].value_counts(normalize = True)[1]*100
        except:
            Pclass = 0.0
        print("Ticket Class "+str(i)+" : "+ str(Pclass)+"%")
    print("----------")

ticket_class_survival_rate(train,3)

### Survival rate against age

## Split into age groups 1-10,11-20,21-30,31-40,41-50,>50

group1 = train["Survived"][train["Age"]<11].value_counts(normalize = True)[1]*100
group2 = train["Survived"][train["Age"]>10][train["Age"]<21].value_counts(normalize = True)[1]*100
group3 = train["Survived"][train["Age"]>20][train["Age"]<31].value_counts(normalize = True)[1]*100
group4 = train["Survived"][train["Age"]>30][train["Age"]<41].value_counts(normalize = True)[1]*100
group5 = train["Survived"][train["Age"]>40][train["Age"]<51].value_counts(normalize = True)[1]*100
group6 = train["Survived"][train["Age"]>50].value_counts(normalize = True)[1]*100

print("Survival rate and Age")
print("----------")
print("Ages 1 - 10 survival rate = " + str(group1)+ "%")
print("Ages 11 - 20 survival rate = " + str(group2)+ "%")
print("Ages 21 - 30 survival rate = " + str(group3)+ "%")
print("Ages 31 - 40 survival rate = " + str(group4)+ "%")
print("Ages 41 - 50 survival rate = " + str(group5)+ "%")
print("Ages above 50 survival rate = " + str(group6)+ "%")
print("----------")

### Survival rate against number of siblings/spouses

_0_sibsp = train["Survived"][train["SibSp"]== 0].value_counts(normalize = True)[1]*100
_1_sibsp = train["Survived"][train["SibSp"]== 1].value_counts(normalize = True)[1]*100
_2_sibsp = train["Survived"][train["SibSp"]== 2].value_counts(normalize = True)[1]*100
_3_sibsp = train["Survived"][train["SibSp"]== 3].value_counts(normalize = True)[1]*100
_4_sibsp = train["Survived"][train["SibSp"]== 4].value_counts(normalize = True)[1]*100
_5_sibsp = 0.0
#_5_sibsp = train["Survived"][train["SibSp"]== 5].value_counts(normalize = True)[1]*100 #error as there are no survivors

print("Survival rate and number of siblings/spouses")
print("----------")
print("No siblings/spouses survival rate = " + str(_0_sibsp)+ "%")
print("1 sibling/spouse survival rate = " + str(_1_sibsp)+ "%")
print("2 siblings/spouses survival rate = " + str(_2_sibsp)+ "%")
print("3 siblings/spouses survival rate = " + str(_3_sibsp)+ "%")
print("4 siblings/spouses survival rate = " + str(_4_sibsp)+ "%")
print("5 siblings/spouses survival rate = " + str(_5_sibsp)+ "%")
print("----------")

### Survival rate against number of parents / children

_0_parch = train["Survived"][train["Parch"]== 0].value_counts(normalize = True)[1]*100
_1_parch = train["Survived"][train["Parch"]== 1].value_counts(normalize = True)[1]*100
_2_parch = train["Survived"][train["Parch"]== 2].value_counts(normalize = True)[1]*100
_3_parch = train["Survived"][train["Parch"]== 3].value_counts(normalize = True)[1]*100
_4_parch = 0.0
#_4_parch = train["Survived"][train["Parch"]== 4].value_counts(normalize = True)[1]*100
_5_parch = train["Survived"][train["Parch"]== 5].value_counts(normalize = True)[1]*100
_6_parch = 0.0
#_6_parch = train["Survived"][train["Parch"]== 6].value_counts(normalize = True)[1]*100

print("Survival rate and number of siblings/spouses")
print("----------")
print("No parents / children survival rate = " + str(_0_parch)+ "%")
print("1 parents / children survival rate = " + str(_1_parch)+ "%")
print("2 parents / children survival rate = " + str(_2_parch)+ "%")
print("3 parents / children survival rate = " + str(_3_parch)+ "%")
print("4 parents / children survival rate = " + str(_4_parch)+ "%")
print("5 parents / children survival rate = " + str(_5_parch)+ "%")
print("6 parents / children survival rate = " + str(_6_parch)+ "%")
print("----------")

### Survival rate against embarked port

## replace c = 1 , q = 2, s = 3

train['Embarked'] = train['Embarked'].map({'C': 1, 'Q': 2, 'S': 3})

## replace 2 missing values with mean
mean = train['Embarked'].mean()
train['Embarked'].fillna(mean,inplace=True)


s = train["Survived"][train["Embarked"]== 3 ].value_counts(normalize = True)[1]*100
c = train["Survived"][train["Embarked"]== 1 ].value_counts(normalize = True)[1]*100
q = train["Survived"][train["Embarked"]== 2 ].value_counts(normalize = True)[1]*100

print("Survival rate and number of siblings/spouses")
print("----------")
print("Cherbourg survival rate = " + str(c)+ "%")
print("Queenstown survival rate = " + str(q)+ "%")
print("Southampton survival rate = " + str(s)+ "%")
print("----------")


## Logistic regression
print("Logistic regression")

X = train.ix[:,(1,2,3,4,5,6,7)].values
y = train.ix[:,0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state=25)

LogReg = LogisticRegression()
LogReg.fit(X_train, y_train)
y_pred = LogReg.predict(X_test)

confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
print(classification_report(y_test, y_pred))

#print(rfe.support_)
#print(rfe.ranking_)

#logit_model=sm.Logit(y,X)
#result=logit_model.fit()
#print(result.summary())


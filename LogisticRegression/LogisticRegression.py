import pandas as pd
import numpy as np
import pickle

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split,KFold,GridSearchCV

from sklearn.metrics import roc_auc_score,confusion_matrix,classification_report


train = pd.read_csv("cleaned_train.csv")
test = pd.read_csv("cleaned_test.csv")

Id = test['PassengerId']

train.drop(columns = ['PassengerId','Name','Ticket','Cabin',],inplace = True)
test.drop(columns = ['PassengerId','Name','Ticket','Cabin',],inplace = True)


X = train[train.columns[train.columns != 'Survived']].copy()
y = train['Survived'].copy()

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=25)


def train_log_reg(train):
    
    X = train[train.columns[train.columns != 'Survived']].copy()
    y = train['Survived'].copy()
    
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=25)

    model = LogisticRegression()
    
    print(" +++ Tuning Hyperparameters +++ ")
    hyperparameters = dict(C = [0.0001, 0.001, 0.01, 1, 100] , penalty = ['l1', 'l2'])
    clf = GridSearchCV(model, hyperparameters, cv=5,scoring='roc_auc',verbose=0)
    logreg = clf.fit(X_train, y_train)
    
    print('Best Parameters : ',clf.best_params_)
    
    print("+++Fitting Logistic regression Model+++")
    
    y_pred = logreg.predict(X_test)
    
    print("roc_auc_score : " , roc_auc_score(y_test,y_pred))
    
    c_matrix = confusion_matrix(y_test, y_pred)
    print(c_matrix)
    print(classification_report(y_test, y_pred))
     
    with open('logreg.pickle','wb') as f:
        pickle.dump(logreg,f)
        
    return logreg

def predict_prob(train,test):
    model = train_log_reg(train)
    pred = model.predict(test)
    
    return pred

pred = predict_prob(train,test)

data_to_submit = pd.DataFrame({'PassengerId':Id,'Survived':pred})
print(data_to_submit.head())
data_to_submit.to_csv('logreg_1.csv', index = False)
    
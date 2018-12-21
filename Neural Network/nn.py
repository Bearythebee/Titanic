# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 16:13:37 2018

@author: Admin
"""

import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split,KFold,GridSearchCV
from sklearn.metrics import roc_auc_score,confusion_matrix,classification_report

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model,Sequential
from keras import initializers, regularizers, constraints, optimizers, layers

train = pd.read_csv("../cleaned_train.csv")
test = pd.read_csv("../cleaned_test.csv")

Id = test['PassengerId']

train.drop(columns = ['PassengerId','Name','Ticket','Cabin',],inplace = True)
test.drop(columns = ['PassengerId','Name','Ticket','Cabin',],inplace = True)


X = train[train.columns[train.columns != 'Survived']].copy()
y = train['Survived'].copy()

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=25)


def train_nn(train):
    
    X = train[train.columns[train.columns != 'Survived']].copy()
    y = train['Survived'].copy()
    
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=25)

    model = Sequential()
    model.add(Dense(12, input_dim=7, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation="sigmoid"))
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    print(" +++ Fitting model +++ ")
    
    model.fit(X_train, y_train, epochs=250, batch_size=5,validation_split=0.1,verbose=2)
    
    y_pred = model.predict(X_test)
    
    print("roc_auc_score : " , roc_auc_score(y_test,y_pred))
     
    model.save("basic_nn_model")
    #with open('nn.pickle','wb') as f:
    #    pickle.dump(model,f)
        
    return model

def predict_prob(train,test):
    model = train_nn(train)
    pred = model.predict(test)
    
    return pred

pred = predict_prob(train,test)



data_to_submit = pd.DataFrame({'PassengerId':Id,'Survived':np.round(pred.flatten()).astype(int)})
print(data_to_submit.head())
data_to_submit.to_csv('nn_1.csv', index = False)
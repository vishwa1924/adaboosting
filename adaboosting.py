# -*- coding: utf-8 -*-
"""
Created on Fri May  8 13:52:32 2020

@author: Sai Viswa
"""


from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

from sklearn import datasets

wine=datasets.load_wine()

X=wine.data

y=wine.target
import pandas as pd
df=pd.read_csv("Social_Network_Ads.csv")

df.size

X=df.iloc[:,[2,3]].values

type(X)

X=df.iloc[:,[2,3]]


type(X)

y=df.iloc[:,4].values

y
from sklearn.model_selection import train_test_split


X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)



X_test.size

from sklearn.ensemble import AdaBoostClassifier

abc=AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),n_estimators=100)



abc.fit(X,y)

abc.score(X,y)



pred=abc.predict(X_test)

abc.score(X_test,pred)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,pred)










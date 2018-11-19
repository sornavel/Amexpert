# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 10:09:29 2018

@author: sorna
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 19:29:12 2018

@author: sorna
"""

#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random as rnd
import seaborn as sns

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from xgboost import XGBClassifier

#importing the dataset
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')


#inspect the columns
train_df.columns
train_df.describe()



# Get one hot encoding of columns B
one_hot = pd.get_dummies(train_df['city_development_index'])
# Drop column B as it is now encoded
train_df = train_df.drop('city_development_index',axis = 1)
# Join the encoded df
train_df = train_df.join(one_hot)

#create new column for day of week
train_df['DateTime'] = pd.to_datetime(train_df['DateTime'], format="%Y-%m-%d %H:%M")
train_df.dtypes
train_df = train_df.reset_index()
train_df['Weekday'] = train_df['DateTime'].dt.dayofweek
train_df.head()

#missing data
total = train_df.isnull().sum().sort_values(ascending=False)
percent = (train_df.isnull().sum()/train_df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)

#drop datetime city_development_index, product_category_2 due to many missing values and relevance
train_df=train_df.drop(['DateTime','session_id', 'user_id','product_category_2','var_1'], axis=1)
train_df.describe()

#missing data
total = train_df.isnull().sum().sort_values(ascending=False)
percent = (train_df.isnull().sum()/train_df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)


train_df=train_df.apply(lambda x:x.fillna(x.value_counts().index[0]))

#missing data
total = train_df.isnull().sum().sort_values(ascending=False)
percent = (train_df.isnull().sum()/train_df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)

train_df['gender'] = train_df['gender'].map( {'Male': 1, 'Female': 2} ).astype(int)
train_df['product'] = train_df['product'].map({'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7,'H':8,'I':9,'J':10,'K':11,'L':12,'M':13,'N':14,'O':15,'P':16,'Q':17,'R':18,'S':19,'T':20,'U':21,'V':22,'W':23,'X':24,'Y':25,'Z':26} ).astype(int)

X_train = train_df.drop("is_click", axis=1)
Y_train = train_df["is_click"]

########################test processing#####################################

# Get one hot encoding of columns B
one_hot = pd.get_dummies(test_df['city_development_index'])
# Drop column B as it is now encoded
test_df = test_df.drop('city_development_index',axis = 1)
# Join the encoded df
test_df = test_df.join(one_hot)

#create new column for day of week
test_df['DateTime'] = pd.to_datetime(test_df['DateTime'], format="%m-%d-%Y %H:%M")
test_df.dtypes
test_df = test_df.reset_index()
test_df['Weekday'] = test_df['DateTime'].dt.dayofweek
test_df.head()


#missing data
total = test_df.isnull().sum().sort_values(ascending=False)
percent = (test_df.isnull().sum()/test_df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)

temp_df = test_df
#drop datetime, city_development_index, product_category_2 due to many missing values
test_df=test_df.drop(['DateTime','session_id', 'user_id','product_category_2'], axis=1)
test_df.describe()

#missing data
total = test_df.isnull().sum().sort_values(ascending=False)
percent = (test_df.isnull().sum()/test_df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)


test_df=test_df.apply(lambda x:x.fillna(x.value_counts().index[0]))

#missing data
total = test_df.isnull().sum().sort_values(ascending=False)
percent = (test_df.isnull().sum()/test_df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)

test_df['gender'] = test_df['gender'].map( {'Male': 1, 'Female': 2} ).astype(int)
test_df['product'] = test_df['product'].map({'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7,'H':8,'I':9,'J':10,'K':11,'L':12,'M':13,'N':14,'O':15,'P':16,'Q':17,'R':18,'S':19,'T':20,'U':21,'V':22,'W':23,'X':24,'Y':25,'Z':26} ).astype(int)

X_test = test_df.drop("is_click", axis=1)
Y_test = test_df["is_click"]

# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_test = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log


# calculate the fpr and tpr for all thresholds of the classification
probs = logreg.predict_proba(X_train)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(Y_train, preds)
roc_auc = metrics.auc(fpr, tpr)

# method I: plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

output_df = pd.DataFrame({"is_click": Y_test,"session_id": temp_df['session_id']})
output_df.to_csv("output_file_dayofweek_cdi.csv")
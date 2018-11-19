# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 12:44:17 2018

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
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn import metrics
from xgboost import XGBClassifier



#importing the dataset
train = pd.read_csv('train.csv')
test= pd.read_csv('test.csv')

#inspect the columns
train_df.columns
train_df.describe()


#create new columns for day of week and hour
train_df['DateTime'] = pd.to_datetime(train_df['DateTime'], format="%Y-%m-%d %H:%M")
train_df = train_df.reset_index()
train_df['Hour'] = train_df['DateTime'].dt.hour
train_df['Weekday'] = train_df['DateTime'].dt.dayofweek
train_df.head()

plt.figure(figsize = (16,5))
df_new=train_df.groupby(['Hour','Weekday'])['is_click'].size()
df_new = df_new.reset_index(name="count")
# Pivot the dataframe to create a [hour x weekday] matrix containing counts
sns.heatmap(df_new.pivot("Hour", "Weekday", "count"), annot=False, cmap="PuBuGn")

#missing data
total = train_df.isnull().sum().sort_values(ascending=False)
percent = (train_df.isnull().sum()/train_df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)

#drop datetime city_development_index, product_category_2 due to many missing values and relevance
train_df=train_df.drop(['DateTime','session_id', 'user_id','product_category_2','city_development_index','var_1'], axis=1)
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
#create new columns for day of week and hour
test_df['DateTime'] = pd.to_datetime(test_df['DateTime'], format="%m-%d-%Y %H:%M")
test_df = test_df.reset_index()
test_df['Hour'] = test_df['DateTime'].dt.hour
test_df['Weekday'] = test_df['DateTime'].dt.dayofweek
test_df.head()

#missing data
total = test_df.isnull().sum().sort_values(ascending=False)
percent = (test_df.isnull().sum()/test_df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)

temp_df = test_df
#drop datetime, city_development_index, product_category_2 due to many missing values
test_df=test_df.drop(['DateTime','session_id', 'user_id','product_category_2','city_development_index'], axis=1)
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
logreg = DecisionTreeClassifier()
logreg.fit(X_train, Y_train)
Y_test_prob = logreg.predict_proba(X_test)
Y_train_prob = logreg.predict_proba(X_train)
Y_train_prob_format = Y_train_prob[:,1]
Y_test = Y_test_prob[:,1]
Y_train = Y_train_prob[:,1]
print ('Train AUC: ',roc_auc_score(Y_train,Y_train_prob))
print ('Test AUC: ',roc_auc_score(Y_test,Y_test_prob))
acc_log = roc_auc_score(Y_train_prob_format, Y_train)
acc_log



est = LogisticRegression()
X = X_train
y = Y_train
est.fit(X, y)

false_positive_rate, true_positive_rate, thresholds = roc_curve(y, est.predict(X))
print (auc(false_positive_rate, true_positive_rate))
# 0.857142857143
print (roc_auc_score(y, est.predict(X)))
# 0.857142857143

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

output_df = pd.DataFrame({"is_click": Y_out,"session_id": temp_df['session_id']})
output_df.to_csv("output_file_final.csv")
import csv
import argparse
import pandas as pd
from datetime import date
from sklearn import datasets,ensemble,metrics
from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor,XGBClassifier

parser = argparse.ArgumentParser()
parser.add_argument("infile")
parser.add_argument("test_data")
args=parser.parse_args()

df = pd.read_csv(args.infile, sep=r'\s*,\s*', header=0, encoding='ascii', engine='python')
df_test = pd.read_csv(args.test_data, sep=r'\s*,\s*', header=0, encoding='ascii', engine='python')



#X_train,X_test,y_train,y_test=train_test_split(df.drop(['adr','is_canceled'],axis=1),df[['adr','is_canceled']],test_size=0.2,random_state=0)

X_train=df.drop(['adr','is_canceled'],axis=1)
y_train=df[['adr','is_canceled']]
X_test=df_test

def to_month_number(row):
	return month_days[int(row['arrival_date_month'])]+row['arrival_date_day_of_month']

def total_n(row):
	return row['stays_in_week_nights']+row['stays_in_weekend_nights']

def total_guest(row):
	return row['adults']+row['children']+row['babies']



X_train['total_night']=X_train.apply(lambda x: total_n(x),axis = 1)
X_test['total_night']=X_test.apply(lambda x: total_n(x),axis = 1)

X_train['total_guests']=X_train.apply(lambda x: total_guest(x),axis = 1)
X_test['total_guests']=X_test.apply(lambda x: total_guest(x),axis = 1)

#X_train=X_train.drop(['arrival_date_day_of_month','arrival_date_year','arrival_date_day_of_month'],axis=1)
#X_test=X_test.drop(['arrival_date_day_of_month','arrival_date_year','arrival_date_day_of_month'],axis=1)
X_train=X_train.drop(['Unnamed: 0'],axis=1)
X_test=X_test.drop(['Unnamed: 0'],axis=1)
missing_cols = set( X_train.columns ) - set( X_test.columns )
for c in missing_cols:
    X_test[c] = 0
X_test = X_test[X_train.columns]
print(X_train.columns)

reg = XGBRegressor(verbosity=1)
clf = ensemble.RandomForestClassifier(n_estimators=100,verbose=1)

reg.fit(X_train,y_train['adr'])
clf.fit(X_train,y_train['is_canceled'])

ypred_adr = reg.predict(X_test)
ypred_can = clf.predict(X_test)


out = pd.DataFrame()
out[['stays_in_week_nights']]=df_test[['stays_in_week_nights']]
out[['stays_in_weekend_nights']]=df_test[['stays_in_weekend_nights']]
out[['arrival_date_year']]=2017
out[['arrival_date_month']]=df_test[['arrival_date_month']]
out[['arrival_date_day_of_month']]=df_test[['arrival_date_day_of_month']]
out[['adr']]=pd.Series(ypred_adr)
out[['is_canceled']]=pd.Series(ypred_can)
out.to_csv('result.csv')

#accuracy_adr = metrics.mean_squared_error(y_test['adr'],ypred_adr)
#accuracy_can = metrics.accuracy_score(y_test['is_canceled'],ypred_can.round())
##print(accuracy_adr)
#print(accuracy_can)
import csv
import argparse
import pandas as pd
from datetime import date
from sklearn import datasets,ensemble,metrics
from sklearn.model_selection import train_test_split
import xgboost as xgb


parser = argparse.ArgumentParser()
parser.add_argument("infile")
parser.add_argument("test_data")
args=parser.parse_args()

df = pd.read_csv(args.infile, sep=r'\s*,\s*', header=0, encoding='ascii', engine='python')
df_test = pd.read_csv(args.test_data, sep=r'\s*,\s*', header=0, encoding='ascii', engine='python')



X_train,X_test,y_train,y_test=train_test_split(df.drop(['adr','is_canceled'],axis=1),df[['adr','is_canceled']],test_size=0.2,random_state=0)

#X_train=df.drop(['adr','is_canceled'],axis=1)
#y_train=df[['adr','is_canceled']]
#X_test=df_test

#missing_cols = set( X_train.columns ) - set( X_test.columns )
#for c in missing_cols:
#    X_test[c] = 0
#X_test = X_test[X_train.columns]
#X_train=X_train.drop(['arrival_date_year'],axis=1)
#X_test=X_test.drop(['arrival_date_year'],axis=1)

dtrain_adr = xgb.DMatrix(X_train,label = y_train['adr'])
dtest_adr = xgb.DMatrix(X_test,label = y_test['adr'])
dtrain_can = xgb.DMatrix(X_train,label = y_train['is_canceled'])
dtest_can = xgb.DMatrix(X_test,label = y_test['is_canceled'])

param_adr = {'max_depth': 6, 'eta': 0.3, 'objective': 'reg:squarederror'}
evallist_adr = [(dtest_adr, 'eval'), (dtrain_adr, 'train')]
param_can = {'max_depth': 6, 'eta': 0.3, 'objective': 'binary:logistic'}
evallist_can = [(dtest_can, 'eval'), (dtrain_can, 'train')]



num_round = 400
adr_model = xgb.train(param_adr, dtrain_adr, num_round, evallist_adr)
can_model = xgb.train(param_can, dtrain_can, num_round, evallist_can)


ypred_adr = adr_model.predict(dtest_adr)
ypred_can = can_model.predict(dtest_can)




#out = pd.DataFrame()
#out[['stays_in_week_nights']]=X_test[['stays_in_week_nights']]
#out[['stays_in_weekend_nights']]=X_test[['stays_in_weekend_nights']]
#out[['arrival_date_year']]=2017
#out[['arrival_date_month']]=X_test[['arrival_date_month']]
#out[['arrival_date_day_of_month']]=X_test[['arrival_date_day_of_month']]
#out[['adr']]=pd.Series(predict_adr)
#out[['is_canceled']]=pd.Series(predict_cancel)
#out.to_csv('result.csv')


#accuracy = metrics.accuracy_score(y_test['is_canceled'],predict_cancel)
accuracy_adr = metrics.mean_squared_error(y_test['adr'],ypred_adr)
accuracy_can = metrics.accuracy_score(y_test['can'],round(ypred_can))
#print(accuracy)
print(accuracy_adr)
print(accuracy_can)
import csv
import argparse
import pandas as pd
from datetime import date
from sklearn import datasets,ensemble,metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier,MLPRegressor
from sklearn.preprocessing import StandardScaler

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

missing_cols = set( X_train.columns ) - set( X_test.columns )
for c in missing_cols:
    X_test[c] = 0
X_test = X_test[X_train.columns]
X_train=X_train.drop(['arrival_date_year'],axis=1)
X_test=X_test.drop(['arrival_date_year'],axis=1)
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

clf = MLPClassifier(solver='adam',learning_rate_init=0.001,alpha=1e-5,hidden_layer_sizes=(20),random_state=1,max_iter=5000,verbose=True)
reg = MLPRegressor(solver='adam',learning_rate_init=0.001,alpha=1e-5,hidden_layer_sizes=(32,16,8),random_state=1,max_iter=5000,verbose=True)


clf.fit(X_train_std,y_train['is_canceled'])
reg.fit(X_train_std,y_train['adr'])
predict_cancel = clf.predict(X_test_std)
predict_adr = reg.predict(X_test_std)
out = pd.DataFrame()
out[['stays_in_week_nights']]=X_test[['stays_in_week_nights']]
out[['stays_in_weekend_nights']]=X_test[['stays_in_weekend_nights']]
out[['arrival_date_year']]=2017
out[['arrival_date_month']]=X_test[['arrival_date_month']]
out[['arrival_date_day_of_month']]=X_test[['arrival_date_day_of_month']]
out[['adr']]=pd.Series(predict_adr)
out[['is_canceled']]=pd.Series(predict_cancel)
out.to_csv('result.csv')
#accuracy = metrics.accuracy_score(y_test['is_canceled'],predict_cancel)
#accuracy_reg = metrics.mean_squared_error(y_test['adr'],predict_adr)
#print(accuracy)
#print(accuracy_reg)
import csv
import argparse
import pandas as pd
from datetime import date
from sklearn import datasets,ensemble,metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

parser = argparse.ArgumentParser()
parser.add_argument("infile")
parser.add_argument("test_data")
#parser.add_argument("outfile")
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
print(X_train.columns)

poly = PolynomialFeatures(2,interaction_only=True)
X_train=poly.fit_transform(X_train)
X_test=poly.fit_transform(X_test)

forest_class = ensemble.RandomForestClassifier(n_estimators=100,verbose=1)
forest_reg = ensemble.RandomForestRegressor(200,verbose=1)

forest_class.fit(X_train,y_train['is_canceled'])
forest_reg.fit(X_train,y_train['adr'])

predict_cancel = forest_class.predict(X_test)
predict_adr = forest_reg.predict(X_test)

out = pd.DataFrame()
out['stays_in_week_nights']=X_test['stays_in_week_nights']
out['stays_in_weekend_nights']=X_test['stays_in_weekend_nights']
out['arrival_date_year']=X_test['arrival_date_year']
out['arrival_date_month']=X_test['arrival_date_month']
out['arrival_date_day_of_month']=X_test['arrival_date_day_of_month']
out['adr']=pd.Series(predict_adr)
out['is_canceled']=pd.Series(predict_cancel)
out.to_csv('result.csv')

#accuracy = metrics.accuracy_score(y_test['is_canceled'],predict_cancel)
#accuracy_reg = metrics.mean_squared_error(y_test['adr'],predict_adr)
#print(accuracy)
#print(accuracy_reg)
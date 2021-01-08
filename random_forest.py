import csv
import argparse
import pandas as pd
from datetime import date
from sklearn import datasets,ensemble,metrics
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument("infile")
args=parser.parse_args()

df = pd.read_csv(args.infile, sep=r'\s*,\s*', header=0, encoding='ascii', engine='python')

X_train,X_test,y_train,y_test=train_test_split(df.drop(['adr','is_canceled'],axis=1),df[['adr','is_canceled']],test_size=0.2,random_state=0)
forest_class = ensemble.RandomForestClassifier(n_estimators=100)
forest_reg = ensemble.RandomForestRegressor(200)

forest_class.fit(X_train,y_train['is_canceled'])
forest_reg.fit(X_train,y_train['adr'])
predict_cancel = forest_class.predict(X_test)
predict_adr = forest_reg.predict(X_test)
accuracy = metrics.accuracy_score(y_test['is_canceled'],predict_cancel)
accuracy_reg = metrics.mean_squared_error(y_test['adr'],predict_adr)
print(accuracy)
print(accuracy_reg)
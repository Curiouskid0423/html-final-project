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
args=parser.parse_args()

df = pd.read_csv(args.infile, sep=r'\s*,\s*', header=0, encoding='ascii', engine='python')


X_train,X_test,y_train,y_test=train_test_split(df.drop(['adr','is_canceled'],axis=1),df[['adr','is_canceled']],test_size=0.2,random_state=0)

sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)

clf = MLPClassifier(solver='adam',alpha=1e-5,hidden_layer_sizes=(20),random_state=1,max_iter=3000)
reg = MLPRegressor(solver='adam',alpha=1e-5,hidden_layer_sizes=(20,5),random_state=1,max_iter=3000)


clf.fit(X_train,y_train['is_canceled'])
reg.fit(X_train,y_train['adr'])
predict_cancel = clf.predict(X_test)
predict_adr = reg.predict(X_test)
accuracy = metrics.accuracy_score(y_test['is_canceled'],predict_cancel)
accuracy_reg = metrics.mean_squared_error(y_test['adr'],predict_adr)
print(accuracy)
print(accuracy_reg)
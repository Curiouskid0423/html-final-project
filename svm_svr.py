import csv
import argparse
import pandas as pd
import re
import time
from datetime import date
from sklearn.svm import SVC,SVR
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

parser = argparse.ArgumentParser()
parser.add_argument("infile")
args=parser.parse_args()

df = pd.read_csv(args.infile, sep=r'\s*,\s*', header=0, encoding='ascii', engine='python')
df = df[:20000]
svm = SVC(kernel='rbf',gamma = 1 ,shrinking = False)
svm_reg = SVR(kernel ='rbf',gamma = 100,shrinking = False,C=0.01,epsilon=0.2)
X_train,X_test,y_train,y_test=train_test_split(df.drop(['adr','is_canceled'],axis=1),df[['adr','is_canceled']],test_size=0.2,random_state=0)

print("Data Preprocessed")
sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)

print("Normalized")

svm.fit(X_train,y_train['is_canceled'].values)
svm_reg.fit(X_train,y_train['adr'].values)

print("Model Training Finished")

pre = svm.predict(X_test)
pre_adr = svm_reg.predict(X_test)
error = 0
error_reg_sqr = 0
#print(pre)
#print(pre_adr)
#print(y_test['adr'].values)
#print(type(pre))
for i,v in enumerate(pre):
    if v!=y_test['is_canceled'].values[i]:
        error+=1
    error_reg_sqr += pow(pre_adr[i]-y_test['adr'].values[i],2)
    #print(str(pre[b])+" "+str(y_test['is_canceled'].values[b]))
print(error/pre.size)
print(error_reg_sqr/pre.size)
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

d_month = {"January":1,
"February":2,
"March":3,
"April":4,
"May":5,
"June":6,
"July":7,
"August":8,
"September":9,
"October":10,
"November":11,
"December":12
}

def first_4(x):
	return int(x[:4])
def mid_2(x):
	return int(x[5:7])
def last_2(x):
	return int(x[8:])

df = pd.read_csv(args.infile, sep=r'\s*,\s*', header=0, encoding='ascii', engine='python')
print("Data Read")
df[['hotel']]=df[['hotel']].apply(lambda x: x=='Resort Hotel').astype(int)
df[['country']]=df[['country']].apply(lambda x: x=='PRT').astype(int)

#df[['reservation_status_date']]=df.apply(lambda x: iso_to_stamp(_x=x['reservation_status_date']))
df[['reserve_year']]=df[['reservation_status_date']].apply(lambda x: x.map(first_4))
df[['reserve_month']]=df[['reservation_status_date']].apply(lambda x: x.map(mid_2))
df[['reserve_date']]=df[['reservation_status_date']].apply(lambda x: x.map(last_2))
df[['arrival_date_month']]=df[['arrival_date_month']].apply(lambda x: x.map(d_month))

df = pd.get_dummies(data=df, columns=['reservation_status','deposit_type','meal', 'market_segment','distribution_channel','reserved_room_type','assigned_room_type','customer_type'])
# df.drop(['meal', 'market_segment','distribution_channel','reserved_room_type','assigned_room_type','customer_type'],axis=1)
label = df[['adr','is_canceled']]
df = df.drop(['adr','is_canceled','reservation_status_date'],axis=1)
df = df.astype(float)
df = df.fillna(0)

svm = SVC(kernel='rbf',gamma = 1 ,shrinking = False)
svm_reg = SVR(kernel ='rbf',gamma = 1,shrinking = False,C=1.0,epsilon=0.2)
X_train,X_test,y_train,y_test=train_test_split(df,label[['is_canceled','adr']],test_size=0.4,random_state=0)

print("Dadta Preprocessed")
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

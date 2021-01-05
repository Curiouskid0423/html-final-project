import csv
import argparse
import pandas as pd
import re
import time
from datetime import date
from sklearn import datasets

parser = argparse.ArgumentParser()
parser.add_argument("infile")
parser.add_argument("outfile")
args=parser.parse_args()


def get_arrival(year,month,date):
	return time.mktime(time.strptime(str(year)+' '+str(month)+' '+str(date),'%Y %B %d'))

def iso_to_stamp(_x):
	DAY = 24*60*60
	a = date.fromisoformat(str(_x))
	return (a-date(1970, 1, 1)).days


df = pd.read_csv(args.infile, sep=r'\s*,\s*', header=0, encoding='ascii', engine='python')
aaa = df[['reservation_status_date']]
print(aaa)
df[['hotel']]=df[['hotel']].apply(lambda x: x=='Resort Hotel').astype(int)
df[['country']]=df[['country']].apply(lambda x: x=='PRT').astype(int)
df[['deposit_type']]=df[['deposit_type']].apply(lambda x: x!='No Deposit').astype(int) #其實有三類
df[['reservation_status']]=df[['reservation_status']].apply(lambda x: x=='Check-Out').astype(int)

df[['arr_date']]=df.apply(lambda x: get_arrival(year = x['arrival_date_year'], month = x['arrival_date_month'], date = x['arrival_date_day_of_month']), axis=1)
df[['reservation_status_date']]=df.apply(lambda x: iso_to_stamp(_x=x['reservation_status_date']), axis=1)

df.drop(['ID','arrival_date_year','arrival_date_month','arrival_date_day_of_month'],axis=1)

df = pd.get_dummies(data=df, columns=['meal', 'market_segment','distribution_channel','reserved_room_type','assigned_room_type','customer_type'])
# df.drop(['meal', 'market_segment','distribution_channel','reserved_room_type','assigned_room_type','customer_type'],axis=1)
label = df[['adr','is_canceled']]
df.drop(['adr','is_canceled'],axis=1)
print(df)
print(label)










import csv
import argparse
import pandas as pd
import re
import time
from datetime import date


parser = argparse.ArgumentParser()
parser.add_argument("infile")
parser.add_argument("outfile")
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
#df[['reserve_year']]=df[['reservation_status_date']].apply(lambda x: x.map(first_4))
#df[['reserve_month']]=df[['reservation_status_date']].apply(lambda x: x.map(mid_2))
#df[['reserve_date']]=df[['reservation_status_date']].apply(lambda x: x.map(last_2))
df[['arrival_date_month']]=df[['arrival_date_month']].apply(lambda x: x.map(d_month))

df = pd.get_dummies(data=df, columns=['deposit_type','meal', 'market_segment','distribution_channel','reserved_room_type','assigned_room_type','customer_type'])
# df.drop(['meal', 'market_segment','distribution_channel','reserved_room_type','assigned_room_type','customer_type'],axis=1)
#label = df[['adr','is_canceled']]
df = df.drop(['ID'],axis=1)
df = df.drop(['reservation_status','reservation_status_date'],axis=1)
df = df.astype(float)
df = df.fillna(0)
df.to_csv(args.outfile)



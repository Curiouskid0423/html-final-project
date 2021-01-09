import csv
import argparse
import pandas as pd
import datetime

parser = argparse.ArgumentParser()
parser.add_argument("infile")
args = parser.parse_args()

def sum(a,b):
    return a+b

data = pd.read_csv(args.infile)
data = data[data["is_canceled"]==0]
data = data.drop(["Unnamed: 0"],axis=1)
numdate = [30,31,30,31,31]
result = pd.read_csv('test_nolabel.csv')
result['label']=0
for i,week_n,weekend_n,year,month,day,adr,cancel in data.itertuples():
    str_date = str(int(year))+'-'+str(int(month)).zfill(2)+'-'+str(int(day)).zfill(2)
    ind=result.index[result['arrival_date'].str.match(str_date)].to_list()[0]
    result.loc[ind,'label']=result.loc[ind,'label']+adr*(weekend_n+week_n)

result['label']=result['label'].apply(lambda x: int(x/10000))

fo = open("outfile.csv","w")

fo.write("arrival_date,label\n")

for i, date,label in result.itertuples():
    fo.write(date+','+str(label)+'\n')
fo.close()


import csv
import argparse
import re
from parse import parse
from datetime import date,strptime

parser = argparse.ArgumentParser()
parser.add_argument("infile")
parser.add_argument("outfile")
parser.parse_args()

with open(args.infile, newline='') as csvfile:
	rows = csv.reader(csvfile)

dict_hotel = {}
dict_month = {}
dict_meal = {}
dict_country = {}
dict_market_segment = {}
dict_distribution_channel = {}
dict_reserved_type = {}
dict_assigned_type = {}
dict_deposit = {}
dict_agent = {}
dict_company = {}
dict_customer_type = {}
dict_reserve_status = {}

def date_to_int(str_date):
	return date.fromisoformat(str_date).timestamp()/86400


def to_num(feature_index,dict_feature):
	count=0
	for row in rows:
		if !(row[feature_index] in dict_feature):
			dict_feature[feature_index]=count
			count++

def numerify(entry):
	newline = []
	count = 0
	newline[count++]=dict_hotel[entry[0]]
	newline[count++]=entry[1]
	newline[count++]=entry[2]
	newline[count++]=datetime.strptime(str(entry[3])+"/"+entry[4]+"/"+str(entry[6]), '%y/%B/%d').timestamp()/86400
	newline[count++]=entry[7]
	newline[count++]=entry[8]
	newline[count++]=entry[9]
	newline[count++]=entry[10]
	newline[count++]=entry[11]
	newline[count++]=dict_meal[entry[12]]
	newline[count++]=dict_country[entry[13]]
	newline[count++]=dict_market_segment[entry[14]]
	newline[count++]=dict_distribution_channel[entry[15]]
	newline[count++]=entry[16]
	newline[count++]=entry[17]
	newline[count++]=entry[18]
	newline[count++]=dict_reserved_type[entry[19]]
	newline[count++]=dict_assigned_type[entry[20]]
	newline[count++]=entry[21]
	newline[count++]=dict_deposit[entry[22]]
	newline[count++]=entry[23]#non-numerical
	newline[count++]=entry[24]#non-numerical
	newline[count++]=entry[25]
	newline[count++]=dict_customer_type[entry[26]]
	newline[count++]=entry[27]
	newline[count++]=entry[28]
	newline[count++]=entry[29]
	newline[count++]=dict_reserve_status[entry[30]]
	newline[count++]=date.fromisoformat(entry[31]).timestamp()/86400
	return newline


map(numerify,rows)













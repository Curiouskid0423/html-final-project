import csv
import argparse
import pandas as pd
from datetime import date
from sklearn import datasets,ensemble,metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier,MLPRegressor

parser = argparse.ArgumentParser()
parser.add_argument("infile")
args=parser.parse_args()

df = pd.read_csv(args.infile, sep=r'\s*,\s*', header=0, encoding='ascii', engine='python')

X_train,X_test,y_train,y_test=train_test_split(df.drop(['adr','is_canceled'],axis=1),df[['adr','is_canceled']],test_size=0.2,random_state=0)

clf = MLPClassifier(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(20,5),random_state=1)
reg = MLPRegressor(random_state=1,max_iter=1000)

clf.fit(X_train,y_train['is_canceled'])
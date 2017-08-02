import csv
import datetime
import decimal
import sys
import pickle
import numpy as np
import pandas as pd
from fbprophet import Prophet
from multiprocessing import Pool
import os


f=open(sys.argv[1])
data=csv.reader(f);

header=next(data)
date_range=header[1:len(header)]
data=[rows for rows in data]

def process(row):
    print row[0]
    filename='models/'+row[0].replace('/','')[-100:]
    if os.path.isfile(filename):
	del filename
	return
    y=[0 for i in xrange(len(date_range))]
    for i in xrange(1,len(row)):
        if row[i]=='' or row[i]=='0':
            y[i-1]=0.000001
        else:
            y[i-1]=float(decimal.Decimal(row[i]))
    df=pd.DataFrame({'ds':date_range,'y':y})
    try:
        m=Prophet()
        m.fit(df)
    except:
   	m=Prophet()
    	m.fit(df,algorithm='Newton')
    future = m.make_future_dataframe(periods=60)
    forecast = m.predict(future)
    filename=open(filename,'wb')
    pickle.dump(forecast[['ds','yhat']],filename,pickle.HIGHEST_PROTOCOL)
    filename.close()
    del forecast
    del m
    del filename
    del y
    del df
    del future

pool=Pool(int(sys.argv[4]))
pool.map(process,data)

start_date=datetime.datetime.strptime(date_range[0],'%Y-%m-%d')
g=open(sys.argv[2])
key=csv.reader(g)
f=open(sys.argv[3],'w')
f.write('Id,Visits\n')
next(key)

j=0;
for row in key:
	print j
	j+=1
	record=row[0].split('_')
	date=(datetime.datetime.strptime(record[len(record)-1],'%Y-%m-%d')-start_date).days
	page='_'.join(record[:len(record)-1])
	forecast=open('models/'+page.replace('/','')[-100:],'rb')
	forecast=pickle.load(forecast)
	f.write(row[1]+','+str(round(forecast.iloc[date]['yhat']))+'\n')

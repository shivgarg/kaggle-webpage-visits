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
import matplotlib.pyplot as plt
import matplotlib

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
    y=np.log(y)	 
    df=pd.DataFrame({'ds':date_range,'y':y})
    
    try:
        m=Prophet(yearly_seasonality=True)
        m.fit(df)
    except:
	try:
	   	m=Prophet(yearly_seasonality=True)
    		m.fit(df,algorithm='Newton')
	except:
		print row[0]
		global model_mean
		model_mean[row[0]]=np.mean(np.exp(y))
		return
    future = m.make_future_dataframe(periods=60)
    forecast = m.predict(future)	
#   fig=m.plot(forecast)
#   plt.show(block=True)
#   fig.savefig('test.png')
    filename1=open(filename,'wb')
    pickle.dump(forecast[['ds','yhat','yhat_lower','yhat_upper']],filename1,pickle.HIGHEST_PROTOCOL)
    filename1.close()
    filename1=open(filename+'.model','wb')
    pickle.dump(m,filename1,pickle.HIGHEST_PROTOCOL)
    filename1.close()

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
	filename= 'models/'+page.replace('/','')[-100:]
	if not os.path.isfile(filename):
		print page
		f.write(row[1]+',0\n')	
	else:
		forecast=open(filename,'rb')
		forecast=pickle.load(forecast)
		f.write(row[1]+','+str(decimal.Decimal(round(np.exp(forecast.iloc[date]['yhat']))))+'\n')

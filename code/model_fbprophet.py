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
    filename='models_60/'+row[0].replace('/','')[-100:]
    if os.path.isfile(filename):
	del filename
	return
    y=[]
    date=[]
    for i in xrange(1,len(row)):
        if row[i]=='0':
            y.append(0.1)
	    date.append(date_range[i-1])
        elif row[i]!='':
            y.append(float(row[i]))
	    date.append(date_range[i-1])
    try:
   	y=np.log(y)	 
    	np.clip(y,np.percentile(y,15),np.percentile(y,75))
    	df=pd.DataFrame({'ds':date,'y':y})
        m=Prophet()
        m.fit(df)
    except:
	try:
	   	m=Prophet()
   		m.fit(df,algorithm='Newton')
	except:
		val = -1
		filename1 = open(filename+'.mean','wb')
		if len(y) !=0 :
			val = np.mean(y)
		pickle.dump(val,filename1,pickle.HIGHEST_PROTOCOL)
		filename1.close()
		return
    future = m.make_future_dataframe(periods=60)
    forecast = m.predict(future)
	
#   print forecast
#   fig=m.plot(forecast)
#   plt.show(block=True)
#   fig.savefig('test.png')

	filename1=open(filename,'wb')
    pickle.dump(forecast,filename1,pickle.HIGHEST_PROTOCOL)
    filename1.close()
    filename1=open(filename+'.model','wb')
    pickle.dump(m,filename1,pickle.HIGHEST_PROTOCOL)
    filename1.close()
    filename1=open(filename+'.date','wb')
    pickle.dump(date,filename1,pickle.HIGHEST_PROTOCOL)
    del forecast
    del m
    del filename
    del y
    del df
    del future

pool=Pool(int(sys.argv[4]))
pool.map(process,data)
pool.close()
pool.join()

start_date=datetime.datetime.strptime('2017-03-01','%Y-%m-%d')
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
	date=(datetime.datetime.strptime(record[len(record)-1],'%Y-%m-%d')-start_date).days - 1
	page='_'.join(record[:len(record)-1])
	filename= 'models_60/'+page.replace('/','')[-100:]
	if not os.path.isfile(filename):
		val=open(filename+'.mean','rb')
		val=pickle.load(val)
		f.write(row[1]+','+str(float(np.exp(val)))+'\n')	
	else:
		forecast=open(filename,'rb')
		forecast=pickle.load(forecast)
		f.write(row[1]+','+str(float(np.exp(forecast.iloc[date]['yhat'])))+'\n')

f.close()

import pandas as pd
import sys
import os
import multiprocessing
import datetime
import pickle
import numpy as np
import numpy
from numpy import log

data=[]
start_date=''
end_date=''
lock=''
page_map=dict()

def read_data(filename):
	global data,start_date,end_date
	data=pd.read_csv(filename).fillna(np.exp(-1.5)).replace(0,np.exp(-1.5))
	start_date=data.keys()[1]
	end_date=data.keys()[-1]
	
def split_page_info():
	global data,page_map
	data['country']=data.iloc[:,0].apply(lambda a: a.split('_')[-3].split('.')[0])
	data['mode']=data.iloc[:,0].apply(lambda a: a.split('_')[-2])
	data['agent']=data.iloc[:,0].apply(lambda a: a.split('_')[-1])

	tmp=dict(zip(data['country'].unique(),range(len(data['country'].unique()))))
	data['country']=data['country'].apply(lambda a: tmp[a])
	f=open("country_map.pickle",'wb')
	pickle.dump(tmp,f)

	tmp=dict(zip(data['mode'].unique(),range(len(data['mode'].unique()))))
	f=open("mode_map.pickle",'wb')
	pickle.dump(tmp,f)
	data['mode']=data['mode'].apply(lambda a: tmp[a])

	tmp=dict(zip(data['agent'].unique(),range(len(data['agent'].unique()))))
	f=open('agent_map.pickle','wb')
	pickle.dump(tmp,f)
	data['agent']=data['agent'].apply(lambda a: tmp[a])

	tmp=dict(zip(data['Page'].unique(),range(len(data['Page'].unique()))))
	page_map=dict(zip(range(len(data['Page'].unique())),data['Page'].unique()))
	f=open('page_map.pickle','wb')
	pickle.dump(tmp,f)
	data['Page']=data['Page'].apply(lambda a: tmp[a])


def expand_data(i):
	global data,start_date,end_date,lock,page_map
	filename=sys.argv[2]
	print i
	tmp=[]

	# Extracting country, mode, agent values 
	page_info=data.iloc[i].filter(['country','mode','agent']).values.tolist()

	# Extracting webpage visits value, taking log and clipping them 
	visits=log(np.array(data.loc[i,start_date:end_date].values))
	np.clip(visits,np.percentile(visits,10),np.percentile(visits,90))
	
	#constructing training samples
	# Country, mode, agenet, day, month, weekday, days offset
	start=datetime.datetime.strptime(start_date,'%Y-%m-%d')
	for j in visits:
		tmp.append(page_info+[j,start.day,start.month,start.weekday(),(start-datetime.datetime(start.year,1,1)).days])
		start+=datetime.timedelta(1)
	
	tmp=pd.DataFrame(tmp,columns=['country','mode','agent','visits','day','month','weekday','offset'])

	# Adding rolling mean and median fields
	tmp['mean_7']=tmp['visits'].shift().rolling(7).mean()
	tmp['mean_30']=tmp['visits'].shift().rolling(30).mean()
	tmp['median_30']=tmp['visits'].shift().rolling(30).median()
	tmp['median_45']=tmp['visits'].shift().rolling(45).median()
	tmp['median_15']=tmp['visits'].shift().rolling(15).median()
	tmp['median_60']=tmp['visits'].shift().rolling(60).median()

	tmp.fillna(-1.5,inplace=True)
	lock.acquire()

	#setting order of columns for csv file
	columns=['country','mode','agent','day','month','weekday','offset','mean_7','mean_30','median_15','median_30','median_45','median_60']
	columns.append('visits')
	if not os.path.isfile(filename):
		tmp.to_csv(filename,index=False,columns=columns)
	else:
		tmp.to_csv(filename, mode='a',header=False,index=False,columns=columns)
	lock.release()

	
	

read_data(sys.argv[1])
split_page_info()
lock=multiprocessing.Lock()
pool=multiprocessing.Pool(int(sys.argv[3]))
pool.map(expand_data,range(len(data)))
pool.close()
pool.join()

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
#	data['Page']=data.iloc[:,0].apply(lambda a: '_'.join(a.split('_'))[:-3])

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


def expand_data(args):
	global data,start_date,end_date,lock,page_map
	filename=sys.argv[2]
	i=args
	print i
	tmp=[]
	page_info=data.iloc[i].filter(['country','mode','agent']).values.tolist()
	visits=log(np.array(data.loc[i,start_date:end_date].values))
	np.clip(visits,np.percentile(visits,10),np.percentile(visits,90))
	start=datetime.datetime.strptime(start_date,'%Y-%m-%d')
	if os.path.isfile('models_60/'+str(page_map[data.iloc[i,0]])):
		prophet_pred=pickle.load(open('models_60/'+page_map[data.iloc[i,0]],'rb'))
		prophet_date=pickle.load(open('models_60/'+page_map[data.iloc[i,0]]+'.date','rb'))
		prophet_pred_ind=0
	else:
		prophet_pred_ind=-1
	for j in visits:
		if (prophet_pred_ind > -1) and (prophet_pred_ind < len(prophet_date)) and (start.strftime('%Y-%m-%d')==prophet_date[prophet_pred_ind]):
			tmp.append(page_info+[j,start.day,start.month,start.weekday(),np.log(prophet_pred.iloc[prophet_pred_ind]['yhat'] if prophet_pred.iloc[prophet_pred_ind]['yhat'] > 0 else np.exp(-2.5))])
			prophet_pred_ind+=1
		else:
			tmp.append(page_info+[j,start.day,start.month,start.weekday(),-2.5])

		start+=datetime.timedelta(1)
	tmp=pd.DataFrame(tmp,columns=['country','mode','agent','visits','day','month','weekday','fb_prophet'])
	tmp['mean_7']=tmp['visits'].shift().rolling(7).mean()
	tmp['mean_30']=tmp['visits'].shift().rolling(30).mean()
	tmp['median_30']=tmp['visits'].shift().rolling(30).median()
	tmp['median_45']=tmp['visits'].shift().rolling(45).median()
#	for i in xrange(7):
#		tmp[str(i)]=tmp['visits'].shift(i+1)
	tmp.fillna(-1.5,inplace=True)
	lock.acquire()
	columns=['country','mode','agent','day','month','weekday','fb_prophet','mean_7','mean_30','median_30','median_45']
#	for i in xrange(7):
#		columns.append(str(i))
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
#arr=[[sys.argv[2],i] for i in xrange(len(data))]
#print arr
pool.map(expand_data,range(len(data)))
pool.close()
pool.join()
#expand_data(sys.argv[2])

#for i in range(len(data)):
#	expand_data(i)

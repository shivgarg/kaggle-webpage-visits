import pandas as pd
import sys
import csv
import datetime
from sklearn.linear_model import ElasticNetCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
import pickle
from numpy import median,mean
from operator import itemgetter
import numpy as np
from xgboost import XGBRegressor
import xgboost as xgb
import os
from keras import backend as K
data=''
page_map=dict()
test_data=''
model=''

def sample_mean_absolute_percentage_error(y_true, y_pred):
        diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true)+K.abs(y_pred), K.epsilon(), None))
        return 200. * K.mean(diff, axis=-1)



def prep_data():
	global data,page_map,test_data
	data=pd.read_csv(sys.argv[1])
	data.fillna(np.exp(-1.5),inplace=True)
	data.replace(0,np.exp(-2.5),inplace=True)
	data=data.values.tolist()
	for i in xrange(len(data)):
		page_map[data[i][0]]=i
	test_data=pd.read_csv(sys.argv[2])
	test_data['pred']=""
	test_data['date']=test_data['Page'].apply(lambda a: datetime.datetime.strptime(a.split('_')[-1],'%Y-%m-%d'))
	test_data.sort_values(by='date',inplace=True)
	test_data['Page']=test_data['Page'].apply(lambda a: '_'.join(a.split('_')[:-1]))


def train():
	global model
	processed_data=pd.read_csv(sys.argv[3])
	processed_data=processed_data.loc[processed_data['visits']!=-1.5]
	Y=processed_data['visits'].values
	columns=['country','mode','agent','day','month','weekday','offset','mean_7','mean_30','median_15','median_30','median_45','median_60']
	param={'objective':'reg:linear','tree_method':'exact','eval_metric':'rmse'}
	X=processed_data.loc[:,columns].values
	X=xgb.DMatrix(X,Y)
	print "read X and Y"
	model=xgb.train(param,X,900)
	

def get_prophet_value(page,index):
	filename='models_60/'+page.replace('/','')[-100:]
	if os.path.isfile(filename):
		forecast=open(filename,'rb')
		forecast=pickle.load(forecast)
		return np.log(forecast.iloc[index]['yhat'] if forecast.iloc[index]['yhat'] > 0 else np.exp(-2.5))
	else:
		return -2.5
	

def test():
	global test_data,data,page_map,model
#	start_date=test_data['date'].min()
	start_date=datetime.datetime(2017,9,11)
	actual_start=test_data['date'].min()
	end_date=test_data['date'].max()

	country_map=open('country_map.pickle','rb')
	country_map=pickle.load(country_map)
	
	agent_map=open('agent_map.pickle','rb')
	agent_map=pickle.load(agent_map)
	
	mode_map=open('mode_map.pickle','rb')
	mode_map=pickle.load(mode_map)
	
	cluster_map=open('page_map.pickle','rb')
	cluster_map=pickle.load(cluster_map)

	ans=pd.DataFrame([],columns=['Id','Visits'])	
	while start_date <= end_date:
		record=start_date
		if start_date < actual_start:
			start_date=actual_start
		print start_date
		tmp=test_data.loc[test_data['date'] == start_date]

		if record < actual_start:
			start_date=record
	
		tmp['country']=tmp.iloc[:,0].apply(lambda a: a.split('_')[-3].split('.')[0])
		tmp['mode']=tmp.iloc[:,0].apply(lambda a: a.split('_')[-2])
		tmp['agent']=tmp.iloc[:,0].apply(lambda a: a.split('_')[-1])
		tmp['country']=tmp['country'].apply(lambda a: country_map[a])

		tmp['mode']=tmp['mode'].apply(lambda a: mode_map[a])

		tmp['agent']=tmp['agent'].apply(lambda a: agent_map[a])

		tmp['day']=start_date.day
		tmp['month']=start_date.month
		tmp['weekday']=start_date.weekday()
		tmp['offset']=(start_date-datetime.datetime(start_date.year,1,1)).days
		tmp['mean_7']=map(lambda a:mean(np.log(a[-7:])),itemgetter(*[page_map[name] for name in tmp['Page']])(data))
		tmp['mean_30']=map(lambda a:mean(np.log(a[-30:])),itemgetter(*[page_map[name] for name in tmp['Page']])(data))
		tmp['median_30']=map(lambda a:median(np.log(a[-30:])),itemgetter(*[page_map[name] for name in tmp['Page']])(data))
		tmp['median_45']=map(lambda a:median(np.log(a[-45:])),itemgetter(*[page_map[name] for name in tmp['Page']])(data))
		tmp['median_15']=map(lambda a:median(np.log(a[-15:])),itemgetter(*[page_map[name] for name in tmp['Page']])(data))
		tmp['median_60']=map(lambda a:median(np.log(a[-60:])),itemgetter(*[page_map[name] for name in tmp['Page']])(data))
		
		columns=['country','mode','agent','day','month','weekday','offset','mean_7','mean_30','median_15','median_30','median_45','median_60']
		tmp.fillna(-1.5,inplace=True)
		pred=model.predict(xgb.DMatrix(tmp.loc[:,columns].values))
		tmp['Visits']=np.exp(pred)
		print pred
		if record >= actual_start:
			ans=ans.append(tmp.loc[:,['Id','Visits']],ignore_index=True)
		ind=0
		for name in tmp['Page']:
			data[page_map[name]].append(np.exp(pred[ind]))
			ind+=1
		start_date+=datetime.timedelta(1)
	ans.to_csv(sys.argv[4],index=False)

print "Training"
train()
print "preparing data"
prep_data()
print "testing"
test()

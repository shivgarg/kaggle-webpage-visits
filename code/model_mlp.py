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
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation
from tensorflow.contrib.keras.python.keras import backend as K


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
	columns=['country','mode','agent','day','month','weekday','mean_7','mean_30','median_30','median_45']
	processed_data=pd.read_csv(sys.argv[3])
	Y=processed_data['visits'].values
	X=processed_data.loc[:,columns].values
	print "read X and Y"
	model=Sequential()
	model.add(Dense(64, input_dim=X.shape[1], activation='linear'))
	model.add(Dropout(0.5))
	model.add(Dense(64, activation='linear'))
	model.add(Dropout(0.5))
	model.add(Dense(1, activation='linear'))
	model.compile(optimizer='rms',
              loss=sample_mean_absolute_percentage_error)

	model.fit(X,Y,epochs=10)

def get_prophet_value(page,index):
	forecast=open('models_60/'+page,'rb')
	forecast=pickle.load(forecast)
	return forecast.iloc[index]['yhat']

def test():
	global test_data,data,page_map,model
	start_date=test_data['date'].min()
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
		print start_date
		tmp=test_data.loc[test_data['date'] == start_date]

		tmp['country']=tmp.iloc[:,0].apply(lambda a: a.split('_')[-3].split('.')[0])
		tmp['mode']=tmp.iloc[:,0].apply(lambda a: a.split('_')[-2])
		tmp['agent']=tmp.iloc[:,0].apply(lambda a: a.split('_')[-1])
		tmp['country']=tmp['country'].apply(lambda a: country_map[a])

		tmp['mode']=tmp['mode'].apply(lambda a: mode_map[a])

		tmp['agent']=tmp['agent'].apply(lambda a: agent_map[a])

		tmp['day']=start_date.day
		tmp['month']=start_date.month
		tmp['weekday']=start_date.weekday()
		tmp['mean_7']=map(lambda a:mean(np.log(a[-7:])),itemgetter(*[page_map[name] for name in tmp['Page']])(data))
		tmp['mean_30']=map(lambda a:mean(np.log(a[-30:])),itemgetter(*[page_map[name] for name in tmp['Page']])(data))
		tmp['median_30']=map(lambda a:median(np.log(a[-30:])),itemgetter(*[page_map[name] for name in tmp['Page']])(data))
		tmp['median_45']=map(lambda a:median(np.log(a[-45:])),itemgetter(*[page_map[name] for name in tmp['Page']])(data))
		
		columns=['country','mode','agent','day','month','weekday','mean_7','mean_30','median_30','median_45']
		pred=model.predict(tmp.loc[:,columns].values)
		pred=np.nan_to_num(pred)
		if len(pred) != len(tmp):
			print "pred"
			print pred
			print "tmp"
			print tmp
			exit(0)
		tmp['Visits']=np.exp(pred)
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

import pandas as pd
import sys
import csv
import datetime
from sklearn.linear_model import ElasticNetCV
import pickle
from numpy import median,mean
from operator import itemgetter

data=''
page_map=dict()
test_data=''
model=''

def prep_data():
	global data,page_map,test_data
	data=pd.read_csv(sys.argv[1])
	data.fillna(0,inplace=True)
	data=data.values.tolist()
	for i in xrange(len(data)):
		page_map[data[i][0]]=i
	print page_map['!vote_en.wikipedia.org_all-access_all-agents']
	test_data=pd.read_csv(sys.argv[2])
	test_data['pred']=""
	test_data['date']=test_data['Page'].apply(lambda a: datetime.datetime.strptime(a.split('_')[-1],'%Y-%m-%d'))
	test_data.sort_values(by='date',inplace=True)
	test_data['Page']=test_data['Page'].apply(lambda a: '_'.join(a.split('_')[:-1]))


def train():
	global model
	processed_data=pd.read_csv(sys.argv[3])
	Y=processed_data['visits'].values
	X=processed_data.iloc[:,:-1].values
	print "read X and Y"
	model=ElasticNetCV(fit_intercept=True,cv=5,copy_X=False,verbose=5,n_jobs=2)
	model.fit(X,Y)


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
#		tmp['Page']=tmp.iloc[:,0].apply(lambda a: '_'.join(a.split('_')[:-3]))

		tmp['country']=tmp['country'].apply(lambda a: country_map[a])

		tmp['mode']=tmp['mode'].apply(lambda a: mode_map[a])

		tmp['agent']=tmp['agent'].apply(lambda a: agent_map[a])

		tmp['day']=start_date.day
		tmp['month']=start_date.month
		tmp['weekday']=start_date.weekday()
		tmp['mean_7']=map(lambda a:mean(a[-7:]),itemgetter(*[page_map[name] for name in tmp['Page']])(data))
		tmp['mean_30']=map(lambda a:mean(a[-30:]),itemgetter(*[page_map[name] for name in tmp['Page']])(data))
		tmp['median_30']=map(lambda a:median(a[-30:]),itemgetter(*[page_map[name] for name in tmp['Page']])(data))
		tmp['median_45']=map(lambda a:median(a[-45:]),itemgetter(*[page_map[name] for name in tmp['Page']])(data))

		pred=model.predict(tmp.loc[:,['country','mode','agent','day','month','weekday','mean_7','mean_30','median_30','median_45']].values)
		tmp['Visits']=pred
		print pred
		ans=ans.append(tmp.loc[:,['Id','Visits']],ignore_index=True)
		ind=0
		for name in tmp['Page']:
			data[page_map[name]].append(pred[ind])
			ind+=1
		start_date+=datetime.timedelta(1)
	ans.to_csv(sys.argv[4],index=False)

print "Training"
train()
print "preparing data"
prep_data()
print "testing"
test()

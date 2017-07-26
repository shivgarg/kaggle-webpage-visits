import sys
import csv
from datetime import datetime, timedelta
from keras.models import Sequential
from keras.layers import Dense
import numpy

f=open(sys.argv[1])
data=csv.reader(f)

header=next(data)
end_date=datetime.strptime(header[len(header)-1],'%Y-%m-%d')
start_date=datetime.strptime(header[0],'%Y-%m-%d')

nns=dict()
search=dict()

for row in data:
	name=row[0].split('_')
	name=''.join(keywords[0:len(name)-3]).replace('/','').replace('File:','')
	search[name]=[0 for i in xrange((end_date-start_date).days)]
	if os.path.isfile('csv/'+name+'.csv'):
		f=open('csv/'+name+'.csv')
		f=csv.reader(f)
		next(f)
		count=0
		for value in f:
			search[name][count]=int(value[1])
			count+=1
	x=[]
	y=[]
	i=0
	data_date=datetime(2015,3,1,0,0,0)
	cur_date=start_date
	while cur_date <= end_date:
		start_idx=(cur_date-timedelta(31)-data_date).days
		x.append(search[name][start_idx, start_idx+30])
		if row[i+1]=='':
			y.append(0)
		else:
			y.append(int(row[i+1]))
		i+=1
		cur_date+=timedelta(1)
	model = Sequential()
	model.add(Dense(12, input_dim=30, init='uniform', activation='relu'))
	model.add(Dense(8, init='uniform', activation='relu'))
	model.add(Dense(1, init='uniform', activation='normal'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	# Fit the model
	model.fit(x, y, nb_epoch=150, batch_size=10,  verbose=2)
	nns[name]=model

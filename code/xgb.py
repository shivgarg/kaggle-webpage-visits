import datetime
#['ru', 'fr', 'en', 'zh', 'de', 'commons', 'www', 'ja', 'es']
#['mobile-web', 'all-access', 'desktop']
#['spider', 'all-agents']
# 49175
# day
# month
# date

import time
import csv
import datetime
from scipy.sparse import csr_matrix,vstack
import numpy as np
import math
import sys
import cPickle as pickle
import gzip
from sklearn.datasets import dump_svmlight_file
import xgboost as xgb

f = open(sys.argv[1])
f = csv.reader(f)
dates = next(f)
data = []
for rows in f:
	data.append(rows)



country = dict()
access = dict()
mode = dict()
pages=dict()
i_country = 0
i_access = 0
i_mode = 0
i_page = 0
for rows in data:
	row=rows[0].split('_')
	domain = row[-3].split('.')[0]
	if not domain in country:
		country[domain]=i_country
		i_country+=1
	if not row[-2] in access:
		access[row[-2]]=i_access
		i_access+=1
	if not row[-1] in mode:
		mode[row[-1]]=i_mode
		i_mode+=1
	name = '_'.join(row[:-3])
	if not name in pages:
		pages[name]=i_page
		i_page+=1

X_dict = dict()
Y_dict = dict()

ind=0
for row in data:
	print ind
	ind+=1	
	domain = [0 for i in xrange(i_country)]
	acc = [0 for i in xrange(i_access)]
	mod = [0 for i in xrange(i_mode)]
	page = [0 for i in xrange(int(math.ceil(math.log(i_page,2))))]

	visits = row[1:]	
	row=row[0].split('_')

	dom = row[-3].split('.')[0]
	domain[country[dom]]=1
	
	acc[access[row[-2]]]=1

	mod[mode[row[-1]]]=1
	
	name = '_'.join(row[:-3])
	index = pages[name]
	k=0
	while index != 0:
		page[k]=index%2
		index=index/2
		k+=1
	tmp_data_x=[]
	tmp_data_y=[]
	for i in xrange(1,len(dates)):
		date=datetime.datetime.strptime(dates[i],'%Y-%m-%d')
		tmp= domain+acc+mod+page+[date.weekday(),date.month,date.day]
		tmp_data_x.append((tmp))
		if visits[i-1]=='':
			tmp_data_y.append(0.0)
		else:
			tmp_data_y.append(float(visits[i-1]))					
	if name in X_dict:
		X_dict[name]=vstack((X_dict[name],csr_matrix(tmp_data_x)))
		Y_dict[name]=Y_dict[name]+tmp_data_y
	else:
		X_dict[name]=csr_matrix(tmp_data_x)
		Y_dict[name]=tmp_data_y

del data
print "dict processing done"
tree_X=[X_dict[name] for name in X_dict]
tree_Y=[Y_dict[name] for name in Y_dict]
print "pickle dump in progress"

with gzip.GzipFile('dict_x.pgz', 'w') as f:
            pickle.dump(tree_X, f)
f.close()
with gzip.GzipFile('dict_y.pgz', 'w') as f:
            pickle.dump(l, f)
f.close()
del X_dict
del Y_dict
num=len(tree_X)

while num!=1:
	print num
	tmp_X=[[] for j in xrange(num/2 if num%2 ==0 else (num+1)/2)]
	tmp_Y=[[] for j in xrange(len(tmp_X))]
	i=0
	while i<num:
		if i+1 < num:
			tmp_X[i/2]=vstack((tree_X[i],tree_X[i+1]))
			tmp_Y[i/2]=tree_Y[i]+tree_y[i+1]
			i+=2
		else:
			tmp_X[i/2]=tree_X[i]
			tmp_Y[i/2]=tree_Y[i]
			i+=1
	tree_X=tmp_X
	tree_Y=tmp_Y		

print "X Y loaded"

X=tree_X[0]
Y=tree_Y[0]
del tree_X
del tree_Y
try:
	dump_svmlight_file(X,Y,"svm_dump")
except:
	print "svm dump failed"


print "dumping X and Y"

with gzip.GzipFile('model_x.pgz', 'w') as f:
            pickle.dump(X, f)
f.close()

with gzip.GzipFile('model_y.pgz', 'w') as f:
            pickle.dump(Y, f)
f.close()



print country.keys()
print access.keys()
print mode.keys()
print len(pages.keys())



import csv
import sys
import datetime
import decimal

f=open(sys.argv[1])
data=csv.reader(f);

next(data)

mean_table = dict()

j=0

for row in data:
	j+=1
	count= [0,0,0,0,0,0,0]
	add=[0,0,0,0,0,0,0]
	day=3
	i=1
	while i <= 550:
		if row[i] != '':
			add[day]+=int(decimal.Decimal(row[i]))
		count[day]+=1
		day+=1
		day%=7
		i+=1
	for i in xrange(7):
		if count[i]!=0:
			add[i]=int(round(float(add[i])/count[i]))
	mean_table[row[0]]=add
	print j

f=open(sys.argv[2])
key=csv.reader(f)
f=open(sys.argv[3],'w')
f.write('Id,Visits\n')
next(key)

j=0;
for row in key:
	print j
	j+=1
	record=row[0].split('_')
	day=datetime.datetime.strptime(record[len(record)-1],'%Y-%m-%d').weekday()	
	f.write(row[1]+','+str(mean_table['_'.join(record[:len(record)-1])][day])+'\n')

f.close()
	

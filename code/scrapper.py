#!/usr/bin/python
# -*- coding: utf-8 -*-

import webbrowser
import time
import os
import shutil
import copy
import pandas as pd
import re
import csv
import numpy as np
from pandas import DataFrame
import sys
import json
import urllib
from datetime import datetime, timedelta
import requests
from functools import reduce

def get_buckets(start_date, end_date):
    start_date_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_date_dt = datetime.strptime(end_date, '%Y-%m-%d')

    bucket_limits = [start_date_dt-timedelta(1)]
    left_limit = start_date_dt
    while left_limit <= end_date_dt:
        new_limit = left_limit + timedelta(days=181)
        if new_limit < end_date_dt:
            bucket_limits.append(new_limit)
        left_limit = new_limit
    bucket_limits.append(end_date_dt)
    return bucket_limits

def get_data(bucket_start_date,bucket_end_date, keyword):
    bucket_start_date_printed = datetime.strftime(bucket_start_date+timedelta(1), '%Y-%m-%d')
    bucket_end_date_printed = datetime.strftime(bucket_end_date, '%Y-%m-%d')
    time_formatted = bucket_start_date_printed + '+' + bucket_end_date_printed
    req = {"comparisonItem":[{"keyword":keyword, "geo":geo, "time": time_formatted}], "category":category,"property":""}
    hl = ""
    tz = "-120"

    explore_URL = 'https://trends.google.com/trends/api/explore?hl={0}&tz={1}&req={2}'.format(hl,tz,json.dumps(req).replace(' ','').replace('+',' '))
    return requests.get(explore_URL).text

def get_token(response_text):
    try:
        return response_text.split('token":"')[1].split('","')[0]
    except:
	print 'CHANGE IP ADDRESS'
   	sys.exit()
        return None

def get_csv_request(response_text):
    try:
        return response_text.split('"widgets":')[1].split(',"lineAnno')[0].split('"request":')[1]       
    except:
        return None

def get_csv(response_text):
    request = get_csv_request(response_text)
    token = get_token(response_text)

    csv = requests.get(u'https://www.google.com/trends/api/widgetdata/multiline/csv?req={0}&token={1}&tz=-120'.format(request,token))
    return csv.text.encode('utf8')

def parse_csv(csv_contents):
    lines = csv_contents.split('\n')
    df = pd.DataFrame(columns = ['date','value'])
    dates = []
    values = []
    # Delete top 3 lines
    for line in lines[3:-1]:
        try:
            dates.append(line.split(',')[0].replace(' ',''))
            values.append(line.split(',')[1].replace(' ',''))
        except:
            pass
    df['date'] = dates
    df['value'] = values
    return df   

def get_daily_frames(start_date, end_date, keyword):

    bucket_list = get_buckets(start_date, end_date)
    frames = []
    for i in range(0,len(bucket_list) - 1):
        resp_text = get_data(bucket_list[i], bucket_list[i+1], keyword)
        frames.append(parse_csv(get_csv(resp_text)))

    return frames

def get_weekly_frame(start_date, end_date, keyword):

    if datetime.strptime(start_date, '%Y-%m-%d') > datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=180):
        print 'No need to stitch; your time interval is short enough. '
        return None
    else:
        resp_text = get_data(datetime.strptime(start_date, '%Y-%m-%d')-timedelta(1), datetime.strptime(end_date, '%Y-%m-%d'), keyword)
        return parse_csv(get_csv(resp_text))

def stitch_frames(daily_frames, weekly_frame,start_date,end_date):

    daily_frame = pd.concat(daily_frames, ignore_index = True)
    daily_frame.columns = ['Date', 'Daily_Volume']
    pd.to_datetime(daily_frame['Date'])
    weekly_frame.columns = ['Week_Start_Date', 'Weekly_Volume']
    daily_frame.index = daily_frame['Date']
    weekly_frame.index = weekly_frame['Week_Start_Date']

    bins = []

    for i in range(0,len(weekly_frame)):
        bins.append(pd.date_range(weekly_frame['Week_Start_Date'][i],periods=7,freq='d'))

    final_data = {}

    for i in range(0,len(bins)):
        week_start_date = datetime.strftime(bins[i][0],'%Y-%m-%d')
	sum_interest = reduce((lambda x, y: float(x) + float(y)), daily_frame['Daily_Volume'].loc[week_start_date:datetime.strftime(bins[i][len(bins[i])-1],'%Y-%m-%d')],0)
        for j in range(0,len(bins[i])):
	    if bins[i][j] > datetime.strptime(end_date,'%Y-%m-%d'):
		break
            this_date = datetime.strftime(bins[i][j],'%Y-%m-%d')
            if sum_interest !=0:
		try:
                	this_val = float(weekly_frame['Weekly_Volume'][week_start_date])*float(daily_frame['Daily_Volume'][this_date])/float(sum_interest)
                	final_data[this_date] = this_val
		except KeyError:
			final_data[this_date]=0
            else:
		final_data[this_date] = 0.0
    
    final_data_frame = DataFrame.from_dict(final_data,orient='index').sort_index()
#   final_data_frame[0] = np.round(final_data_frame[0]/final_data_frame[0].max()*100,2)

    final_data_frame.columns=['Volume']
    final_data_frame.index.names = ['Date']

    final_data_frame.to_csv('csv/{0}.csv'.format(keywords.replace('+','').replace('/','')), sep=',')

                                                                                                                                                                                        
f=open(sys.argv[1])
f=csv.reader(f)
next(f)
for row in f:
    keywords=row[0].split('_')
    filename='csv/'+''.join(keywords[0:len(keywords)-3]).replace('/','').replace('File:','').replace('.jpg','').replace('User:','')+'.csv'
    if os.path.isfile(filename):
	continue
    keywords='+'.join(keywords[0:len(keywords)-3]).replace('File:','').replace('.jpg','').replace('User:','')
    start_date = '2015-03-01'
    end_date = '2017-07-20'
    geo = ''
    category = 0
    print keywords
    daily_frames = get_daily_frames(start_date, end_date, keywords)
    weekly_frame = get_weekly_frame(start_date, end_date, keywords)
   
    empty=weekly_frame.empty
    if not empty:
	    empty=True     
	    for frame in daily_frames:
		if not frame.empty:
			empty=False
    if not empty:
	print "stitching"
    	stitch_frames(daily_frames, weekly_frame,start_date,end_date)
    else:
	print "Writing empty file"
	f=open('csv/'+keywords.replace('+','').replace('/','')+'.csv','w')
	f.close()

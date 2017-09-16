import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm, tqdm_notebook
import time
import requests
import pickle

SLEEP_TIME_S = 0.1

def extract_URL_and_Name(page):
	return (['_'.join(page.split('_')[:-3])]+ ['http://' + page.split("_")[-3:-2][0] +'/wiki/' + '_'.join(page.split('_')[:-3])])

train = pd.read_csv('../data/train_1.csv')


page_data = pd.DataFrame(list(train['Page'].apply(extract_URL_and_Name)),columns=['Name', 'URL'])


def fetch_wikipedia_text_content(row):
	try:
		r = requests.get(row['URL'])
		time.sleep(SLEEP_TIME_S)
		to_return = [x.get_text() for x in BeautifulSoup(r.content, "html.parser").find(id="mw-content-text").find_all('p')]
	except:
		to_return = [""]
	return to_return

tqdm.pandas(tqdm_notebook)
page_data['TextData'] = page_data.progress_apply(fetch_wikipedia_text_content, axis=1)

f=open('wiki_data','wb')
pickle.dump(page_data,f)
f.close()



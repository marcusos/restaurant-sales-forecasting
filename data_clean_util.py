from nltk.corpus import stopwords
from pickle import dump
import string
import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer
import numpy
import datetime as dt
from brazilian_calendar import get_holidays

# python -c 'from data_clean_util import *; build_clean_dataset()'
def build_clean_dataset():
	# Read sales
	sales = pd.read_csv('data/sales.csv')
	sales['date'] = pd.to_datetime(sales['date'])
	
	# Read macro economy data
	economy = pd.read_csv('data/macro_economy.csv')
	economy['year_month_eco'] = economy.apply(lambda x: str(int(x.year)) + str(int(x.month)).zfill(2) , axis=1)
	
	# Read whaeter data
	dateparse = lambda x: dt.datetime.strptime(x, '%d/%m/%Y')
	wheater = pd.read_csv('data/wheater.csv', parse_dates=['date'], date_parser=dateparse)
	
	# The wheater data rows are duplicated per day, fixing it:
	wheater_min = wheater[wheater.type == 'end']
	wheater = wheater[wheater.type == 'start']
	wheater.reset_index(drop=True)
	wheater.min_temp = wheater_min.min_temp.values
	wheater.precipitation_vol = wheater_min.precipitation_vol.values

	# Generate a date series from 2016-06-20 to 2018-11-29
	base = dt.datetime(2018, 11, 29)
	date_list = [base - dt.timedelta(days=x) for x in range(0, 893)]
	df = pd.DataFrame(date_list, columns=['date'])
	df['year_month'] = df.date.apply(lambda x: x.strftime('%Y%m'))

	# Add holidays data
	holidays = get_holidays(dt.datetime(2016, 6, 20), dt.datetime(2018, 11, 29))
	holidays_df = pd.DataFrame(holidays, columns=['date'])
	holidays_df['holiday'] = int(1)

	# Merge dataframes
	df = pd.merge(df, economy, left_on='year_month', right_on='year_month_eco', how='left')
	df = pd.merge(df, wheater, left_on='date', right_on='date', how='left')
	df = pd.merge(df, sales, left_on='date', right_on='date', how='left')
	df = pd.merge(df, holidays_df, left_on='date', right_on='date', how='left')

	# Some adjustments on week day
	df['day'] = df.date.apply(lambda x: x.day)
	df['week_day'] = df.date.apply(lambda x: x.weekday())
	df['week_day_str'] = df.date.apply(lambda x: x.strftime('%A'))
	
	# Some adjustments on holiday
	df.holiday.fillna(0, inplace=True)
	df['after_holiday'] = df.holiday.shift(-1)
	df['before_holiday'] = df.holiday.shift(1)
	df.after_holiday.fillna(0, inplace=True)
	df.before_holiday.fillna(0, inplace=True)
	# Cast to int
	df.holiday = df.holiday.astype(int)
	df.after_holiday = df.after_holiday.astype(int)
	df.before_holiday = df.before_holiday.astype(int)

	# Drop unused columns 
	df = df.drop(columns=['year_month_eco', 'type', 'station'])

	#df.to_csv('processed/sales.csv', encoding='utf-8')
	df.to_csv('processed/sales.csv')

	#Holiday


	'''
	print("Sales")
	print(sales.describe())
	print(sales.dtypes)
	print("\nEconomy")
	print(economy.describe())
	print(economy.dtypes)
	print("\nWheater")
	print(wheater.describe())
	print(wheater.dtypes)
	print(date_list[0])
	'''
	




# python -c 'from data_clean_util import *; get_dataframes_re()'
# python -c 'import numpy as np; l = np.array([131, 3226, 62, 186]); print(l/l.sum()*100); print(l/l.sum()*100-100);'
# REFERENCES: https://github.com/keras-team/keras/issues/741
def get_dataframes_re():
	df = pd.read_csv('wheater_data.csv')
	print("All classes")
	print(df.groupby(['label_str']).size())
	print("-------------------")
	filter_idxs =  df['label_str'].isin(['negativo_maioria','negativo_unânime','positivo_maioria','positivo_unânime'])
	df = df[filter_idxs]
	df.label_str = pd.Categorical(df.label_str)
	df['label'] = df.label_str.cat.codes
	df['doc'] = df['10']
	df = df[['doc', 'label', 'label_str']]
	#print(df[['10','label_str']].head())
	print("Filtered")
	print(df.head)
	print(df.groupby(['label']).size())

	df['doc'] = df['doc'].map(clean_doc)
	labels = df.label.unique()
	trains = []
	tests = []
	for label in labels:
		train, test = train_test_split(df[df.label == label], test_size=0.2)
		tests.append(test)
		trains.append(train)

	train = pd.concat(trains, keys=labels)
	test= pd.concat(tests, keys=labels)
	print('Splitted: ')
	print(test.groupby(['label_str']).size())
	print(train.groupby(['label_str']).size()) 

	# Get Y
	binarizer = LabelBinarizer()
	trainY = binarizer.fit_transform(train.label_str)
	#trainY = multilabel_binarizer.classes_
	binarizer = LabelBinarizer()
	testY = binarizer.fit_transform(test.label_str)
	#testY = multilabel_binarizer.classes_
	print('multilabel_binarizer example: ')
	#print(testY[0:5])
	# Get X 
	trainX = train['doc'].tolist()
	testX = test['doc'].tolist()
	# Save processed data
	save_dataset([trainX,trainY], 'processed_data/train.pkl')
	save_dataset([testX,testY], 'processed_data/test.pkl')

#MAIN
#df = pd.read_csv('mc_acordoes_data.csv')
# python -c "import data_clean_util;  clean_acordoes_data"
def clean_acordoes_data():
	sqlite_con = sqlite3.connect('../../legaltech/data/stf_acordaos_db')
	df_p, df_n = get_dataframes()
	df_p['doc']= df_p['decisao'].map(clean_doc)
	df_n['doc']= df_n['decisao'].map(clean_doc)

	train_p, test_p = train_test_split(df_p, test_size=0.1)
	train_n, test_n = train_test_split(df_n, test_size=0.1)

	# load all training reviews
	trainX = train_p['doc'].tolist() + train_n['doc'].tolist()
	trainY = train_p['label'].tolist() + train_n['label'].tolist()
	save_dataset([trainX,trainY], 'train.pkl')

	# load all test reviews
	testX = test_p['doc'].tolist() + test_n['doc'].tolist()
	testY = test_p['label'].tolist() + test_n['label'].tolist()
	save_dataset([testX,testY], 'test.pkl')


# Reference data https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/

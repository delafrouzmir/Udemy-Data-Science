# Module
import pickle
import numpy as np 
import pandas as pd 
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# sklrearn's scaler will scale all data,
# we only want to scale numerical (and not categorical) data to scale
class CustomScaler (BaseEstimator, TransformerMixin):
	'''
	A class for custom scaling of data
	by getting x and a subset of its columns,
	we only scale the data in those columns
	and the rest (categorical data/dummy variable) untouched
	'''
	def __init__ (self, columns, copy=True, with_mean = True, with_std = True):
		self.scaler = StandardScaler(copy, with_mean, with_std)
		self.columns = columns
		self.mean = None
		self.std = None

	def fit (self, x, y=None):
		# calculating the mean and std of every desired column of x
		self.scaler.fit(x[self.columns], y)
		self.mean = np.mean(x[self.columns])
		self.std = np.std(x[self.columns])
		return self

	def transform (self, x, y=None, copy=None):
		# standardization of x in desired columns
		# and returning the new x, with some columns scaled
		# and other columns not touched
		original_col_order = x.columns
		x_scaled = self.scaler.transform (x[self.columns], y)
		x_scaled = pd.DataFrame (columns = self.columns, data = x_scaled)
		x_unscaled = x.loc[:, ~x.columns.isin(self.columns)]
		allx = pd.concat ([x_scaled, x_unscaled], axis = 1)
		return allx[original_col_order]


class AbsenteeismModel:
	'''
	Using a previously trained model on employee's absenteeism data
	using logistic regression
	to find out if an employee is likely to be absent
	more than 4 hours (=median of absents) from work in a day
	under special circumstances or not
	'''
	def __init__ (self, scaler_file, model_file):
		# loadng the previously trained model and scaler
		with open(scaler_file, 'rb') as scaler_file, open(model_file, 'rb') as model_file:
			self.scaler = pickle.load(scaler_file)
			self.model = pickle.load(model_file)
			self.data = None

	def load_and_clean (self, data_file):
		self.raw_data = pd.read_csv(data_file, delimiter=',')

		df = self.raw_data.copy()
		self.df_with_predictions = df.copy()

		# creating grouped dummies for 'Reason for Absence' column
		reasons_dummies = pd.get_dummies(df['Reason for Absence'], drop_first = True)
		
		reason1 = reasons_dummies.loc[:,'1':'14'].sum(axis=1)
		reason2 = reasons_dummies.loc[:,'15':'17'].sum(axis=1)
		reason3 = reasons_dummies.loc[:,'18':'21'].sum(axis=1)
		reason4 = reasons_dummies.loc[:,22:28].sum(axis=1)

		df = df.drop (['ID','Reason for Absence'], axis=1)

		data_concat_with_dummies = pd.concat([df, reason1, reason2, reason3, reason4], axis=1)
		
		data_concat_with_dummies.columns = ['Date', 'Transportation Expense',
			'Distance to Work', 'Age', 'Daily Work Load Average','Body Mass Index',
			'Education', 'Children', 'Pets',
			'reason_1', 'reason_2', 'reason_3', 'reason_4']

		# reordering the data
		columns_reordered = ['reason_1', 'reason_2', 'reason_3', 'reason_4',
			'Date', 'Transportation Expense', 'Distance to Work', 'Age',
			'Daily Work Load Average','Body Mass Index',
			'Education', 'Children', 'Pets']

		data_reordered = data_concat_with_dummies[columns_reordered]

		data_checkpointed = data_reordered.copy()
		data_checkpointed['Date'] = pd.to_datetime(data_checkpointed['Date'], format='%d/%m/%Y')
		
		# Extracting day of the week and month from date column
		months = []

		for i in range(0, len(data_checkpointed)):
			months.append(data_checkpointed['Date'][i].month)

		def date_to_weekday (date):
			return date.weekday()

		weekdays = data_checkpointed['Date'].apply(date_to_weekday)

		data_checkpointed = data_checkpointed.drop(['Date'], axis = 1)
		data_checkpointed['Month'] = months
		data_checkpointed['Weekday'] = weekdays

		columns_reordered = ['reason_1', 'reason_2', 'reason_3', 'reason_4',
		       'Transportation Expense', 'Distance to Work', 'Age', 'Daily Work Load Average',
		       'Body Mass Index', 'Education', 'Children', 'Pets',
		       'Month', 'Weekday']
		data_checkpointed = data_checkpointed[columns_reordered]
		data_with_date = data_checkpointed.copy()

		# converting the education:
		# 0: high school degree
		# 1: higher education degree
		data_with_date['Education'] = data_with_date['Education'].map({1:0, 2:1, 3:1, 4:1})

		# Final Checkpoint
		data_preprocessed = data_with_date.copy()

		# We learned from training the model that these features
		# do not play a big role in predicting the target.
		# In Logistic Regression language, they have weights of close to 0
		data_preprocessed = data_preprocessed.drop(['Daily Work Load Average',
				'Distance to Work', 'Month'], axis=1)

		self.preprocessed_data = data_preprocessed.copy()
		
		self.data = self.scaler.transform(data_preprocessed)

	def predicted_probability (self):
		# calculating the probability of extreme absenteeism from work
		if self.data is not None:
			prob = self.model.predict_proba(self.data)
			pred = self.model.predict_proba(self.data)[:,1]
			return pred

	def predicted_category (self):
		# calculating the category of extreme absenteeism from work, 0 or 1
		if self.data is not None:
			pred = self.model.predict(self.data)
			return pred 

	def predicted_output (self):
		# calculating both probability and category
		# and concatenating it to the input data
		if self.data is not None:
			prob = self.predicted_probability()
			cat = self.predicted_category()
			self.preprocessed_data['Probability'] = prob
			self.preprocessed_data['Category'] = cat
			return self.preprocessed_data

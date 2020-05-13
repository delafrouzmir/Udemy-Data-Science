import numpy as np 
import pandas as pd 
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

class CustomScaler (BaseEstimator, TransformerMixin):
	def __init__ (self, columns, copy=True, with_mean = True, with_std = True):
		self.scaler = StandardScaler(copy, with_mean, with_std)
		self.columns = columns
		self.mean = None
		self.std = None

	def fit (self, x, y=None):
		self.scaler.fit(x[self.columns], y)
		self.mean = np.mean(x[self.columns])
		self.std = np.std(x[self.columns])
		return self

	def transform (self, x, y=None, copy=None):
		original_col_order = x.columns
		x_scaled = self.scaler.transform (x[self.columns], y)
		x_scaled = pd.DataFrame (columns = self.columns, data = x_scaled)
		x_unscaled = x.loc[:, ~x.columns.isin(self.columns)]
		allx = pd.concat ([x_scaled, x_unscaled], axis = 1)
		return allx[original_col_order]

data_preprocessed = pd.read_csv('Absenteeism_preprocessed.csv')
#print(data_preprocessed.head())

median = data_preprocessed['Absenteeism Time in Hours'].median()
targets = np.where(data_preprocessed['Absenteeism Time in Hours'] > median, 1, 0)
print('Perentage of excessive Absenteeism is {0:0.2f}'.format(targets.sum()/len(targets) * 100))

data_preprocessed = data_preprocessed.drop(['Absenteeism Time in Hours',
							'Daily Work Load Average', 'Distance to Work', 'Month'], axis=1)
# data_preprocessed = data_preprocessed.drop(['Absenteeism Time in Hours'], axis=1)
data_preprocessed['Excessive Absenteeism'] = targets

data = data_preprocessed.copy()
#print(data.head())

unscaled_inputs = data.iloc[:,:-1]
targets = data.iloc[:,-1]

# Scaling
print(unscaled_inputs.columns)
dummy_columns = ['reason_1', 'reason_2', 'reason_3', 'reason_4', 'Education']

non_dummies = [x for x in unscaled_inputs.columns if x not in dummy_columns]

scaler = CustomScaler(non_dummies)
scaler.fit(unscaled_inputs)
inputs = scaler.transform(unscaled_inputs)

#train test splitting
x_train, x_test, y_train, y_test = train_test_split(inputs,
	targets, train_size=0.8, shuffle=True, random_state = 18)

# model
reg = LogisticRegression()
reg.fit(x_train, y_train)
print(reg.score(x_train, y_train))

# Summary table for model
table = pd.DataFrame(columns=['Features'], data = unscaled_inputs.columns.values)
table['Coeffs'] = np.transpose (reg.coef_)
table.index = table.index + 1
table.loc[0] = ['Intercept',reg.intercept_[0]]
table = table.sort_index()
table['Odds_ratio'] = np.exp(table['Coeffs'])
table = table.sort_values('Odds_ratio', ascending=False)
print(table)

print (reg.score(x_test, y_test))
predicted_prob = reg.predict_proba(x_test)


# saving the model
import pickle

with open('model', 'wb') as file:
	pickle.dump (reg, file)

with open('scaler', 'wb') as file:
	pickle.dump (scaler, file)
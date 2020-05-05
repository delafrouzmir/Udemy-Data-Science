import numpy as np 
import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

def rsqr(x, y, reg) -> [float]:
	rsq = reg.score(x,y)
	n = x.shape[0]
	p = x.shape[1]
	adjr = 1 - (1-rsq)*(n-1)/(n-p-1)
	return [rsq,adjr]


data = pd.read_csv('real_estate_price_size_year2.csv')
print(data.head())

x = data[['size','year']]
y = data['price']

scaler = StandardScaler()
scaler.fit(x)
x_sc = scaler.transform(x)

reg = LinearRegression()
model = reg.fit(x_sc, y)

table = pd.DataFrame(data=['Bias',x.columns[0], x.columns[1]], columns=['Features'])
table['Weights'] = [reg.intercept_, reg.coef_[0], reg.coef_[1]]
print(table)

[r_sq, adjr] = rsqr(x_sc, y, reg)
print('R-squared is {0:.3f} and Adjusted R-squared is: {1:.3f}'.format(r_sq, adjr))

predic_data = pd.DataFrame(data=[[750,2009]], columns=['size','year'])
predic_data_sc = scaler.transform(predic_data)

predic_data['PredictedValue'] = reg.predict(predic_data_sc)
print(predic_data)
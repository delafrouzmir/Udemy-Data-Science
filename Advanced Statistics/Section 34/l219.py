import numpy as np 
import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression

def adjusted_rsqrd (x, y, reg: 'LinearRegression') -> float:
	model = reg.fit(x,y)
	n = x.shape[0]
	p = x.shape[1]
	rsq = reg.score(x,y)

	adjr = 1 - (1-rsq)*(n-1)/(n-p-1)
	print('R-squared is: ', rsq)
	print('Adjusted R-squared is: ', adjr)
	return adjr

data = pd.read_csv('real_estate_price_size_year.csv')
print(data.head())
x = data[['size','year']]
y = data['price']

reg = LinearRegression()
model = reg.fit(x,y)
print('Model\'s coeffs are: ', model.coef_)
print('Model\'s intercept is: ', model.intercept_)

p_values = f_regression(x,y)[1].round(3)
table = pd.DataFrame( data=x.columns.values, columns=['Features'])
table['Coefficients'] = model.coef_
table['P_values'] = p_values
print(table)

adjr = adjusted_rsqrd(x,y,reg)

prediction_data = pd.DataFrame (data=[[750, 2009]], columns=['Size','Year'])
prediction_data['PredictedPrice'] = reg.predict(prediction_data)
print(prediction_data)
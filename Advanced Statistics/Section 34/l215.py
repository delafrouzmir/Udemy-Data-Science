import numpy as np 
import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression

def adjustedRSqrd (x, y, reg: 'LinearRegression') -> float:
	model = reg.fit(x,y)

	r_sq = reg.score(x,y)
	n = x.shape[0]
	p = x.shape[1]

	adj_r = 1 - (1-r_sq)*(n-1)/(n-p-1)
	print('R-squared is: ', r_sq)
	print('Adjusted R-squared is: ' ,adj_r)
	return adj_r

data = pd.read_csv('1.02. Multiple linear regression.csv')
print(data.head())

x = data[['SAT','Rand 1,2,3']]
y = data['GPA']

reg = LinearRegression()
adjr = adjustedRSqrd (x,y,reg)

# summary table of the model containing variables, their coeffs, and p_values
p_values = f_regression(x,y)[1].round(3)
table = pd.DataFrame( data=x.columns.values, columns =['Features'])
table['Coefficients'] = reg.coef_
table['P_values'] = p_values
print(table)
import numpy as np 
import pandas as pd 
import statsmodels.api as statm 

data = pd.read_csv('real_estate_price_size_year_view.csv')
# change the 'sea view' to 1 and 'no sea view' to 0
# since sea view pribably adds to the price
data['view'] = data['view'].map({'No sea view':0, 'Sea view':1})

y = data['price']
x1 = data[['size','year','view']]
x = statm.add_constant(x1)
print(x)

regres = statm.OLS(y,x).fit()
print(regres.summary())

# predict some prices!

houses = pd.DataFrame({'const':1, 'size':[1100,1400,1200,1300], 'year':[2010,1990,1992,2009], 'view':[1,0,1,0]})
price_prediction = regres.predict(houses)

price_predict = pd.DataFrame ({'Prediction':price_prediction})
all_table = houses.join(price_predict)
print(all_table)
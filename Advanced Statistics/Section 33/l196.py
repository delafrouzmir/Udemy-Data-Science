import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import statsmodels.api as statm 
import seaborn as sbn 
sbn.set()

data = pd.read_csv('real_estate_price_size_year.csv')
y = data['price']
x1_1var = data['size']
x_1var = statm.add_constant(x1_1var)
x1_2var = data[['size','year']]
x_2var = statm.add_constant(x1_2var)

regres1var = statm.OLS(y,x_1var).fit()
regres2var = statm.OLS(y,x_2var).fit()
print(regres1var.summary())
print(regres2var.summary())

# both R-squared and Adj R-squared are more
# when we use both variables (size and year)
# which means the model based on the two variables
# is more explanatory than just one variable (size)
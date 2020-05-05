import numpy as np 
import pandas as pd 
import statsmodels.api as statm 
import matplotlib.pyplot as plt  
import seaborn as sbn 
sbn.set()

data = pd.read_csv ('real_estate_price_size.csv')
print(data.describe())

y = data['price']
x1 = data['size']
x = statm.add_constant(x1)

# plt.scatter(x1,y)
# plt.xlabel('size',fontsize=20)
# plt.ylabel('price', fontsize=20)
# plt.show()

regres = statm.OLS(y,x).fit()
print(regres.summary())
print('-----------')
print(regres.params)

plt.scatter(x1,y)
yhat = regres.params['size'] * x1 + regres.params['const']
fig = plt.plot(x1, yhat, lw=4, c='green', label='regression line')
plt.xlabel('size',fontsize=20)
plt.ylabel('price', fontsize=20)
plt.show()

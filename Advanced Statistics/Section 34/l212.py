import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sbn 
sbn.set()

from sklearn.linear_model import LinearRegression

data = pd.read_csv('real_estate_price_size.csv')
print (data.head())
x = data['size']
y = data['price']
x = x.values.reshape(-1,1)
print(x.shape)

reg = LinearRegression()
model = reg.fit(x,y)

# coefs:
print(model.coef_)
# intercept
print(model.intercept_)
# r-squares
print(model.score(x,y))

# plot
plt.scatter(x,y)
yhat = x*model.coef_[0] + model.intercept_
fig = plt.plot(x, yhat, lw=2, c='orange', label='Regression Line')
plt.xlabel('size of property')
plt.ylabel('price of property')
plt.show()

# prediction: price of a house of 750 sqft
pred_data = pd.DataFrame(data=[750], columns=['size'])
pred_data['predicted price'] = pd.DataFrame(reg.predict (pred_data), columns=['price'])
print(pred_data)

import numpy as np 
import pandas as pd 
import statsmodels.api as stm 
import matplotlib.pyplot as plt 

data = pd.read_csv('Bank-data.csv')
print(data.head())

y = data['y'].map({'no':0, 'yes':1})
x = data.drop(['Unnamed: 0','y'], axis=1)
x = stm.add_constant(x)

log_reg = stm.Logit(y,x)
model = log_reg.fit()
print(model.summary())

##########
# regression obly based on duration
##########
x1 = data['duration']
x2 = stm.add_constant(x1)

log_reg = stm.Logit(y,x2)
model = log_reg.fit()
print(model.summary())

def logistic_f (x, b0, b1):
	return np.array( np.exp(b0+b1*x) / (1 + np.exp(b0+b1*x)) )

x_sorted = np.sort(np.array(x1))
y_sorted = np.sort (logistic_f(x1,model.params[0],model.params[1]))

plt.scatter(x1, y, color='navy')
plt.xlabel('duration')
plt.ylabel('yes')
plt.scatter(x_sorted,y_sorted, color='red')
plt.show()
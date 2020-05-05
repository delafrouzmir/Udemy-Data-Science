import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import statsmodels.api as stm 

data = pd.read_csv('Example-bank-data.csv')
print(data.head())

x1 = data['duration']
y = data['y'].map({'no':0, 'yes':1})
x = stm.add_constant(x1)

# reg_log = sm.Logit(y,x)
# results_log = reg_log.fit()

# def f(x,b0,b1):
#     return np.array(np.exp(b0+x*b1) / (1 + np.exp(b0+x*b1)))

# f_sorted = np.sort(f(x1,results_log.params[0],results_log.params[1]))
# x_sorted = np.sort(np.array(x1))

# plt.scatter(x1,y,color='C0')
# plt.xlabel('SAT', fontsize = 20)
# plt.ylabel('Admitted', fontsize = 20)
# plt.plot(x_sorted,f_sorted,color='C8')
# plt.show()

log_reg = stm.Logit(y,x)
model = log_reg.fit()
print(model.summary())

def logistic_f (x, b0, b1):
	return np.array( (np.exp(b0+b1*x))/(1+ np.exp(b0+b1*x)) )

yhat = logistic_f(x1, model.params[0], model.params[1])
x_sorted = np.sort(np.array(x1))
y_sorted = np.sort(yhat)

plt.scatter(x1,y, color='navy')
plt.xlabel('Duration')
plt.ylabel('Subscription')
plt.plot(x_sorted,y_sorted,color='red')
plt.show()
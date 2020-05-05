import numpy as np 
import pandas as pd 
import statsmodels.api as sm 
import matplotlib.pyplot as plt 

raw_data = pd.read_csv('Bank-data.csv')
print(raw_data.head())

data = raw_data.copy()
data = data.drop(['Unnamed: 0'], axis = 1)
y = data['y'].map({'no':0, 'yes':1})
x = data.drop(['y'], axis = 1)
x = sm.add_constant(x)

log_reg = sm.Logit(y,x)
model = log_reg.fit()

print(model.summary())
print(model.params)

print('Coef of duration is: ', model.params[-1])
print('which means with similar situation, one more day of duration increases the subcription odds {0} times'.format( np.exp(model.params[-1])))

###############
# only duration
###############

x2 = data['duration']
x2 = sm.add_constant(x2)

log_reg2 = sm.Logit(y,x2)
model2 = log_reg2.fit()

print(model2.summary())
print(model2.params)

print('Coef of duration is: ', model2.params[-1])
print('when only duration is considered, one more day of duration increases the subcription odds {0} times'.format( np.exp(model2.params[-1])))

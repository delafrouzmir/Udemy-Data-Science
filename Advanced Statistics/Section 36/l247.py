import numpy as np 
import pandas as pd 
import statsmodels.api as sm 

raw_data = pd.read_csv('Bank-data.csv')
data = raw_data.copy().drop(['Unnamed: 0'], axis = 1)

x1 = data.drop(['y'], axis = 1)
x = sm.add_constant(x1)
y = data['y'].map({'no':0, 'yes':1})

log_reg = sm.Logit(y,x)
model = log_reg.fit()

predicted_vals = model.predict(x)
print(model.summary())

def confusion_matrix (actual_data, predicted_data):
	bins = np.array([0,0.5,1])
	cm = np.histogram2d(actual_data, predicted_data, bins=bins)[0]
	return cm

cm = confusion_matrix (y, predicted_vals)
cm_df = pd.DataFrame( data = cm, columns=['Predicted 0','Predicted 1'])
cm_df = cm_df.rename(index = {0: 'Actual 0', 1: 'Actual 1'})
print (cm_df)

accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
print('Accuracy of our model is: ', accuracy)
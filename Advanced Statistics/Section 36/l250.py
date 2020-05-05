import numpy as np 
import pandas as pd 
import statsmodels.api as sm 

raw_train = pd.read_csv('Bank-data.csv')
raw_test = pd.read_csv('Bank-data-testing.csv')

train_data = raw_train.drop(['Unnamed: 0'], axis=1)
test_data = raw_test.drop(['Unnamed: 0'], axis=1)

x1_train = train_data.drop(['y'], axis=1)
y_train = train_data['y'].map({'no':0, 'yes':1})
x_train = sm.add_constant(x1_train)

x1_test = test_data.drop(['y'], axis=1)
y_test = test_data['y'].map({'no':0, 'yes':1})
x_test = sm.add_constant(x1_test)
# making sure test data is in the same order as train data
x_test = x_test[x_train.columns.values]

log_reg = sm.Logit(y_train, x_train)
model = log_reg.fit()
print(model.summary())

def confusion_matrix (test_data, actual_y, model):
	predicted_vals = model.predict(test_data)
	bins = np.array([0,0.5,1])
	cm = np.histogram2d(actual_y, predicted_vals, bins=bins)[0]
	accuracy = (cm[0,0]+cm[1,1])/cm.sum()

	return cm, accuracy

def cmToDataFrame (cm):
	cm_df = pd.DataFrame (data = cm[0], columns=['Predicted 0', 'Predicted 1'])
	cm_df = cm_df.rename (index = {0: 'Actual 0', 1: 'Actual 1'})
	return cm_df

cm = confusion_matrix(x_test, y_test, model)
print ( cm )
print ( cmToDataFrame(cm) )

# building another model without may because it's p-value is more than 0.05
new_x_train = x_train.drop(['may'], axis=1)
new_x_test = x_test.drop(['may'], axis=1)

log_reg2 = sm.Logit(y_train, new_x_train)
model2 = log_reg2.fit()

cm2 = confusion_matrix(new_x_test, y_test, model2)
print('---------------------------------')
print(cm2)
print (cmToDataFrame(cm2))

# accuracy droped, first model is better in accuracy
import pandas as pd 

raw_data = pd.read_csv('Absenteeism-data.csv')
#print(raw_data.head())
#print(raw_data.info())
#print(raw_data.columns)

data = raw_data.copy()

reasons_dummies = pd.get_dummies(data['Reason for Absence'])
check_reasons_dummies = reasons_dummies.sum(axis=1)
#print(check_reasons_dummies.sum())
#print(check_reasons_dummies.unique())

reasons_dummies = pd.get_dummies(data['Reason for Absence'], drop_first = True)

reason1 = reasons_dummies.loc[:,'1':'14'].max(axis=1)
reason2 = reasons_dummies.loc[:,'15':'17'].max(axis=1)
reason3 = reasons_dummies.loc[:,'18':'21'].max(axis=1)
reason4 = reasons_dummies.loc[:,22:28].max(axis=1)

data = data.drop (['ID','Reason for Absence'], axis=1)

data_concat_with_dummies = pd.concat([data, reason1, reason2, reason3, reason4], axis=1)

print(data_concat_with_dummies.columns)
data_concat_with_dummies.columns = ['Date', 'Transportation Expense',
					'Distance to Work', 'Age', 'Daily Work Load Average','Body Mass Index',
					'Education', 'Children', 'Pets', 'Absenteeism Time in Hours',
					'reason_1', 'reason_2', 'reason_3', 'reason_4']

# reordering the data
columns_reordered = ['reason_1', 'reason_2', 'reason_3', 'reason_4',
					'Date', 'Transportation Expense', 'Distance to Work', 'Age',
					'Daily Work Load Average','Body Mass Index',
					'Education', 'Children', 'Pets', 'Absenteeism Time in Hours']

data_reordered = data_concat_with_dummies[columns_reordered]
#print(data_reordered.head())

# Date
data_checkpointed = data_reordered.copy()
#print(type(data_checkpointed['Date'][0]))

data_checkpointed['Date'] = pd.to_datetime(data_checkpointed['Date'], format='%d/%m/%Y')
#print(data_checkpointed.info())

# Extracting day and month
months = []

for i in range(0, len(data_checkpointed)):
	#weekdays.append(data_checkpointed['Date'][i].weekday())
	months.append(data_checkpointed['Date'][i].month)

## Alternatively
def date_to_weekday (date):
	return date.weekday()

weekdays = data_checkpointed['Date'].apply(date_to_weekday)

data_checkpointed = data_checkpointed.drop(['Date'], axis = 1)
data_checkpointed['Month'] = months
data_checkpointed['Weekday'] = weekdays

print(data_checkpointed.columns)
columns_reordered = ['reason_1', 'reason_2', 'reason_3', 'reason_4',
       'Transportation Expense', 'Distance to Work', 'Age', 'Daily Work Load Average',
       'Body Mass Index', 'Education', 'Children', 'Pets',
       'Month', 'Weekday', 'Absenteeism Time in Hours']
data_checkpointed = data_checkpointed[columns_reordered]
data_with_date = data_checkpointed.copy()

# converting the education
print(data_with_date['Education'].unique())
data_with_date['Education'] = data_with_date['Education'].map({1:0, 2:1, 3:1, 4:1})

# Final Checkpoint
data_preprocessed = data_with_date.copy()
print(data_preprocessed.info())
data_preprocessed.to_csv('Absenteeism_preprocessed.csv', index=False)
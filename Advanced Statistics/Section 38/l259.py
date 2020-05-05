import numpy as np 
import pandas as pd 
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt 
import seaborn as sn 
sn.set()

raw_data = pd.read_csv('Categorical.csv')
print(raw_data.head())
print(raw_data['continent'].unique())

x = raw_data.drop(['name'], axis=1)
x['continent'] = x['continent'].map({'North America':0, 'Asia': 1, 'Africa': 2, 'Europe': 3,
	'South America': 4, 'Oceania': 5, 'Antarctica': 6, 'Seven seas (open ocean)': 7})
model = KMeans(4)
model.fit(x)

clusters = model.fit_predict(x)
data = raw_data.iloc[:,1:3]
data['clusters'] = clusters

plt.scatter(data['Longitude'], data['Latitude'], c=data['clusters'], cmap='rainbow')
plt.xlim(-180, 180)
plt.ylim(-90, 90)
plt.show()
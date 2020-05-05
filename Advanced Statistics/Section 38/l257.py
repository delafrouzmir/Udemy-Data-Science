import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans

raw_data = pd.read_csv('Countries-exercise.csv')
print(raw_data.head())

x = raw_data.iloc[:,1:3]

model = KMeans(2)
model.fit(x)

clusters = model.fit_predict(x)
data = x.copy()
data['clusters'] = clusters

plt.scatter(data['Longitude'], data['Latitude'], c=data['clusters'])
plt.xlim(-180,180)
plt.ylim(-90,90)
plt.show()

#################
# with 7 clusters

model2 = KMeans(6)
model2.fit(x)

clusters2 = model2.fit_predict(x)
data2 = x.copy()
data2['clusters'] = clusters2

plt.scatter(data2['Longitude'], data2['Latitude'], c=data2['clusters'])
plt.xlim(-180,180)
plt.ylim(-90,90)
plt.show()
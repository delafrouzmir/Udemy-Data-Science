import numpy as np 
import pandas as pd 
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt 
import seaborn as sn 
sn.set()

raw_data = pd.read_csv('Countries-exercise.csv')
x = raw_data.iloc[:,1:3]

wcss = []
for i in range(1,11):
	model = KMeans(i)
	model.fit(x)
	wcss.append(model.inertia_)

plt.scatter(range(1,11), wcss)
plt.show()

model = KMeans(3)
model.fit(x)

plt.scatter(x['Longitude'], x['Latitude'], c=model.fit_predict(x), cmap='rainbow')
plt.xlim(-180,180)
plt.ylim(-90,90)
plt.show()
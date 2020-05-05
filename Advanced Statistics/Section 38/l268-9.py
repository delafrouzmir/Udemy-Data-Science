import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans
from sklearn import preprocessing
import seaborn as sn 
sn.set()

raw_data = pd.read_csv('iris-dataset.csv')
print(raw_data.head())

x_1 = raw_data.drop(['petal_length','petal_width'], axis=1)
kmeans2 = KMeans(2)
kmeans2.fit(x_1)
clusters_x1 = kmeans2.fit_predict(x_1)

# plt.scatter(x_1['sepal_length'], x_1['sepal_width'], c=clusters_x1, cmap='rainbow')
# plt.show()

x_2 = preprocessing.scale(x_1.copy())
kmeans2_2 = KMeans(2)
kmeans2_2.fit(x_2)
clusters_x2 = kmeans2_2.fit_predict(x_2)

# plt.scatter(x_1['sepal_length'], x_1['sepal_width'], c=clusters_x2, cmap='rainbow')
# plt.show()

##############
# elbow method
wcss = []
for i in range(1,15):
	model = KMeans(i)
	model.fit(x_2)
	wcss.append(model.inertia_)

plt.scatter (range(1,15), wcss)
plt.show()

###############
answers = pd.read_csv('iris-with-answers.csv')
petal_types = answers.iloc[:,-1].unique()
actual_y = answers.iloc[:,-1].map({petal_types[0]:0, petal_types[1]:1, petal_types[2]:2})
print(actual_y)

num_cl = [2,3,4,5]
for i in num_cl:
	model = KMeans(i)
	model.fit(x_2)
	clusters = model.fit_predict(x_2)
	plt.scatter(x_1['sepal_length'], x_1['sepal_width'], c=clusters, cmap='rainbow')
	plt.show()
	print('the accuracy for {0} clusters is {1}%'.format(i, sum(clusters==actual_y)/sum(clusters)*100))
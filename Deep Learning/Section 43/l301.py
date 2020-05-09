import numpy as np 
import matplotlib.pyplot as plt 

observations = 100000
size = np.random.uniform (low=-10, high=10,size = (observations,1))
mile = np.random.uniform (low=-10, high=10, size = (observations,1))

inputs = np.column_stack((size,mile))
noise = np.random.uniform (low=-1, high=1, size=(observations,1))

targets = 3*size - 2*mile + 5 + noise

init_range = 0.1
ws = np.random.uniform(low=-init_range, high=init_range, size=(2, 1))
bs = np.random.uniform(low=-init_range, high=init_range, size=1)

learning_rate = 0.05
for i in range(100):
	ys = np.dot (inputs, ws) + bs
	deltas = ys - targets

	#loss = np.sum ( deltas**2 ) / 2 / observations
	loss = np.sum ( np.absolute(deltas) ) / 2 / observations
	print(loss)
	if loss < 0.001:
		print('Hooray!')
		break

	deltas = deltas / observations
	ws = ws - learning_rate * (np.dot(inputs.T, deltas))
	bs = bs - learning_rate * np.sum(deltas)

print(ws)
print(bs)
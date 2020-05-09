import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 

observations = 100000
input_size = 2
output_size = 1

size = np.random.uniform (low=-10, high=10,size = (observations,1))
mile = np.random.uniform (low=-10, high=10, size = (observations,1))

inputs = np.column_stack((size,mile))
noise = np.random.uniform (low=-1, high=1, size=(observations,1))

targets = 3*size - 2*mile + 5 + noise

np.savez('TF_first', inputs = inputs, targets = targets)

training_data = np.load('TF_first.npz')

model = tf.keras.Sequential([
			tf.keras.layers.Dense (output_size,
                            	kernel_initializer = tf.random_uniform_initializer(minval=0.1, maxval=0.1),
                                bias_initializer = tf.random_uniform_initializer(minval=0.1, maxval=0.1)
                                )
							])

custom_optimizer = tf.keras.optimizers.SGD(learning_rate=0.02)
model.compile (optimizer=custom_optimizer, loss='huber_loss')

model.fit (training_data['inputs'], training_data['targets'], epochs=100, verbose=2)

ws = model.layers[0].get_weights()[0]
bs = model.layers[0].get_weights()[1]

print(ws)
print(bs)

ys = model.predict_on_batch(training_data['inputs'])

plt.scatter (ys, targets)
plt.xlabel('model\'s outputs')
plt.ylabel('real targets')
plt.show()
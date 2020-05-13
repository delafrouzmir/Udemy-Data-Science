import numpy as np 
import tensorflow as tf 

data = np.load('Audiobook_train_data.npz')
train_ins,  train_outs = data['inputs'].astype(np.float), data['targets'].astype(np.int)

data = np.load('Audiobook_validation_data.npz')
validation_ins,  validation_outs = data['inputs'].astype(np.float), data['targets'].astype(np.int)

data = np.load('Audiobook_test_data.npz')
test_ins,  test_outs = data['inputs'].astype(np.float), data['targets'].astype(np.int)

print(train_ins.shape)
print(train_outs.shape)


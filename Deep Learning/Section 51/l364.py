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

input_size = train_ins.shape[1]
output_size = 2
hidden_size = 200

model = tf.keras.Sequential([
                            tf.keras.layers.Flatten(),
                            tf.keras.layers.Dense(hidden_size, activation='relu'),
                            tf.keras.layers.Dense(hidden_size, activation='relu'),
                            tf.keras.layers.Dense(hidden_size, activation='relu'),
                            tf.keras.layers.Dense(output_size, activation='softmax'),
                            ])


model.compile (optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

num_epochs = 50
batch_size = 200
early_stopping = tf.keras.callbacks.EarlyStopping(patience = 2)

model.fit(train_ins, train_outs, batch_size = batch_size, callbacks=[early_stopping],
         validation_data = (validation_ins, validation_outs), epochs = num_epochs, verbose = 2)

test_outs = test_outs.reshape(-1,1)
print(test_outs.shape)

test_loss, test_accuracy = model.evaluate(test_ins, test_outs)
# RESULT:
# (448, 1)
# 448/448 [==============================] - 0s 658us/sample - loss: 0.3717 - accuracy: 0.8058
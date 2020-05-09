import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


##################
# loading the data
##################

dataset, mnist_info = tfds.load(name='mnist', with_info = True, as_supervised = True)
print (mnist_info)

mnist_train, mnist_test = dataset['train'], dataset['test']

num_validation_ex = mnist_info.splits['train'].num_examples // 10
num_validation_ex = tf.cast (num_validation_ex, tf.int64)

num_test_ex = mnist_info.splits['test'].num_examples
num_test_ex = tf.cast (num_test_ex, tf.int64)

##################
#  prepricessing #
##################

def scale (image, label):
    image = tf.cast (image, tf.float32)
    image /= 255.0
    return image, label

scaled_train_validation = mnist_train.map(scale)
scaled_test = mnist_test.map(scale)

# shuffling
BUFFER_SIZE = 10000

shuffled_train_validation = scaled_train_validation.shuffle(BUFFER_SIZE)

# selecting validation data from the train dataset
validation_data = shuffled_train_validation.take(num_validation_ex)
train_data = shuffled_train_validation.skip(num_validation_ex)

# batching: validation and test data can be one batch
BATCH_SIZE = 100

train_data = train_data.batch(BATCH_SIZE)
validation_data = validation_data.batch(num_validation_ex)
test_data = scaled_test.batch(num_test_ex)

# getting validation inputs and targets
validation_inputs, validation_targets = next(iter(validation_data))

# the model
input_size = 28*28
output_size = 10
hidden_size = 300

model = tf.keras.Sequential ([
                                tf.keras.layers.Flatten (input_shape=(28, 28, 1)),
                                tf.keras.layers.Dense ( hidden_size, activation='relu' ),
                                tf.keras.layers.Dense ( hidden_size, activation='tanh' ),
                                tf.keras.layers.Dense ( output_size, activation='softmax' )
                            ])

# optimizer and loss
model.compile (optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# train: max result: 99.73% accuracy on validation set
NUM_EPOCHS = 16

model.fit (train_data, epochs = NUM_EPOCHS, validation_data=(validation_inputs,
			validation_targets), verbose=2, validation_steps=20)

# Testing
test_loss, test_accuracy = model.evaluate (test_data)

print ('Test loss is {0:.3f} and test accuracy is {1:.2f}'.format(test_loss, test_accuracy*100))
# Test loss is 0.089 and test accuracy is 98.32
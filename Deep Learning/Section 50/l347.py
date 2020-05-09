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
BUFFER_SIZE = 20000

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

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

dataset, mnist_info = tfds.load(name='mnist', with_info = True, as_supervised = True)
print (mnist_info)

mnist_train, mnist_test = dataset['train'], dataset['test']

num_validation_ex = mnist_info.splits['train'].num_examples // 10
num_validation_ex = tf.cast (num_validation_ex, tf.int64)

num_test_ex = mnist_info.splits['test'].num_examples
num_test_ex = tf.cast (num_test_ex, tf.int64)

def scale (image, label):
    image = tf.cast (image, tf.float32)
    image /= 255.0
    return image, label

scaled_train_validation = mnist_train.map(scale)
scaled_test = mnist_test.map(scale)
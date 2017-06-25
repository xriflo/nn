'''
A Convolutional Network implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function
from pylab import *

import argparse
import sys
import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True, reshape=False)


LEARNING_RATE = 0.001
NO_EPOCHS = 1001
BATCH_SIZE = 128
BATCH_SIZE_TEST = 1000
DROPOUT = 0.75


# Network Parameters
n_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units

IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
NO_OUTPUT_CLASSES = 10

# tf Graph input
x = tf.placeholder(tf.float32, [None, IMAGE_WIDTH, IMAGE_HEIGHT ,1])
y = tf.placeholder(tf.float32, [None, NO_OUTPUT_CLASSES])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)


def eval_in_batches(data, sess, correct_prediction, train_step):
    """Get all predictions for a dataset by running it in small batches."""
    size = data.images.shape[0]
    no_iters = size/BATCH_SIZE_TEST
    all_sum = 0
    acc = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
    for i in range(no_iters):
      batch_xs, batch_ys = data.next_batch(BATCH_SIZE_TEST)
      curr_acc = sess.run(acc, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})
      all_sum = all_sum + curr_acc
    return all_sum/size


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    #x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out


def main(_):
  # Store layers weight & bias
  weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
  }

  biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
  }



  # Construct model
  pred = conv_net(x, weights, biases, keep_prob)

  # Define loss and optimizer
  cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
  train_step = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)


  #Settings
  train_acc = []
  test_acc = []
  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  
  correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Train
  for i in range(NO_EPOCHS):

    batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)

    _, acc = sess.run([train_step, accuracy], feed_dict={x: batch_xs, y: batch_ys, keep_prob: DROPOUT})
    if i%200==0:
      tr_acc = eval_in_batches(mnist.train, sess, correct_prediction, pred)
      ts_acc = eval_in_batches(mnist.test, sess, correct_prediction, pred)
      print(tr_acc, " vs ", ts_acc)
      train_acc.append(tr_acc)
      test_acc.append(ts_acc)
  print("---------------------------------------------")
  print(train_acc)
  print(test_acc)
  plot ( arange(0,NO_EPOCHS+1,200),train_acc,color='g',label='train acc' )
  plot ( arange(0,NO_EPOCHS+1,200),test_acc,color='r',label='test acc' )
  xlabel('No epochs')
  ylabel('Accuracy')
  title('CNN - training and testing')
  legend(('train acc','test acc'))
  savefig("cnn.png")


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

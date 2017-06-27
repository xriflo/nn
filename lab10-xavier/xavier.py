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
import math

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
    print("x: ",x.shape)
    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    print("conv1: ",conv1.shape)
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)
    print("conv1: ",conv1.shape)
    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    print("conv2: ",conv2.shape)
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)
    print("conv2: ",conv2.shape)
    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    print("fc1: ",fc1.shape)
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    print("fc1: ",fc1.shape)
    fc1 = tf.nn.relu(fc1)
    print("fc1: ",fc1.shape)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)
    print("fc1: ",fc1.shape)
    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    print("out: ",out.shape)
    return out

    '''
    x:  (?, 28, 28, 1)
    conv1:  (?, 28, 28, 32)
    conv1:  (?, 14, 14, 32)
    conv2:  (?, 14, 14, 64)
    conv2:  (?, 7, 7, 64)
    fc1:  (?, 3136)
    fc1:  (?, 1024)
    fc1:  (?, 1024)
    fc1:  (?, 1024)
    out:  (?, 10)
    
    '''


def main(_):
  # Store layers weight & bias
  #no_fan_in  = no_input_feature_maps * filterH * filterW
  #no_fan_out = no_output_feature_maps * filterH * filterW / (poolsizeH * poolsizeW) 
  
  no_fan_in_wc1 = 1*5*5
  no_fan_out_wc1 = 32*5*5/(2*2)
  no_fan_in_wc2 = 32*5*5
  no_fan_out_wc2 = 64*5*5/(2*2)
  no_fan_in_wd1 = 3136
  no_fan_out_wd1 = 1024
  no_fan_in_out = 1024
  no_fan_out_out = 10
  
  std_wc1 = math.sqrt(6.0/(no_fan_in_wc1+no_fan_out_wc1))
  std_wc2 = math.sqrt(6.0/(no_fan_in_wc2+no_fan_out_wc2))
  std_wd1 = math.sqrt(6.0/(no_fan_in_wd1+no_fan_out_wd1))
  std_out = math.sqrt(6.0/(no_fan_in_out+no_fan_out_out))


  weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_uniform([5, 5, 1, 32], -std_wc1, std_wc1)),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_uniform([5, 5, 32, 64], -std_wc2, std_wc2)),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_uniform([7*7*64, 1024], -std_wd1, std_wd1)),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_uniform([1024, n_classes], -std_out, std_out))
  }

  biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
  }



  # Construct model


  # Define loss and optimizer
  
  optimizers_name = ["Adam", "RMSProp", "Adagrad", "Adadelta"]
  optimizers = [
                tf.train.AdamOptimizer(learning_rate=LEARNING_RATE), 
                tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE),
                tf.train.AdagradOptimizer(learning_rate=LEARNING_RATE),
                tf.train.AdadeltaOptimizer(learning_rate=LEARNING_RATE),
                ]
  optimizers_train_values = [[], [], [], []]
  optimizers_test_values = [[], [], [], []]
  colors = ['b', 'g', 'r', 'y']


  #Settings
  train_acc = []
  test_acc = []
  
  weights_copy = []
  biases_copy = []

  #tf.Variable(weights.initialized_value(), name="w2")

  for opt in range(len(optimizers)):
    new_hash_weights = {}
    for key, value in weights.iteritems():
      new_hash_weights[key] = tf.Variable(value.initialized_value)
    weights_copy.append(new_hash_weights)

    new_hash_biases = {}
    for key, value in biases.iteritems():
      new_hash_biases[key] = tf.Variable(value.initialized_value)
    biases_copy.append(new_hash_biases)


    # Train
  for opt in range(len(optimizers)):
    
    pred = conv_net(x, weights_copy[opt], biases_copy[opt], keep_prob)

    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
    print("*------->",optimizers[opt])
    train_step = optimizers[opt].minimize(cost)

    

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    for i in range(NO_EPOCHS):

      batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)

      _, acc = sess.run([train_step, accuracy], feed_dict={x: batch_xs, y: batch_ys, keep_prob: DROPOUT})
      if i%200==0:
        print("----->",i)
        tr_acc = eval_in_batches(mnist.train, sess, correct_prediction, pred)
        ts_acc = eval_in_batches(mnist.test, sess, correct_prediction, pred)
        optimizers_train_values[opt].append(tr_acc)
        optimizers_test_values[opt].append(ts_acc)
  
  print(optimizers_train_values)
  print("------------------------")
  print(optimizers_test_values)

  for opt in range(len(optimizers)):
    plot(arange(0, NO_EPOCHS+1, 200), optimizers_test_values[opt], color=colors[opt], label=optimizers_name[opt])
  xlabel('No epochs')
  ylabel('Accuracy')
  title('Optimizers with Xavier - testing')
  legend(tuple(optimizers_name))
  savefig("xavier.png")
  

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

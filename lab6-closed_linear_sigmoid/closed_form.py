from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from pylab import *

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import scipy as np

FLAGS = None

LEARNING_RATE = 0.04
NO_EPOCHS = 400
BATCH_SIZE = 100

def main(_):

  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # Create the model
  x = tf.Variable(tf.zeros([55000, 784]))
  W = tf.Variable(tf.zeros([785, 10]))
  w = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10]))
  # Define linear classifier
  y = tf.matmul(x, w) + b
  y_ = tf.Variable(tf.zeros([55000, 10]))

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  


  all_batch_xs, all_batch_ys = mnist.train.images, mnist.train.labels

  no_samples = all_batch_xs.shape[0]
  no_features = all_batch_xs.shape[1]

  pad_xs = tf.Variable(tf.ones([no_samples,1]))
  tilda_x = tf.concat([all_batch_xs, pad_xs], 1)
  tilda_x_transp = tf.transpose(tilda_x)
  tilda_xtx = tf.matmul(tilda_x_transp, tilda_x)

  result = tf.py_func(np.linalg.pinv, [tilda_xtx], [tf.float32])
  print(result[0].eval())
  #tilda_x_pinv = tf.matmul(tf.matrix_inverse(result), tilda_x_transp)
  #print(tilda_x_pinv.shape)
  #W = tf.matmul(tf.cast(tilda_x_pinv, tf.float64), all_batch_ys)
'''
  index = tf.range(W.shape[0]-1)
  w = tf.gather(W, index)
  b = tf.gather(W, [W.get_shape().as_list()[0]-1])[0]



  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

 

  #print(tf.reduce_mean(w).eval())

  #print(tf.reduce_sum(w).eval())

  train_acc = sess.run(accuracy, feed_dict={x: mnist.train.images, y_:mnist.train.labels})
  test_acc  = sess.run(accuracy, feed_dict={x: mnist.test.images, y_:mnist.test.labels})

  
  #print("train_acc: ", train_acc, "   test_acc: ", test_acc)
'''



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

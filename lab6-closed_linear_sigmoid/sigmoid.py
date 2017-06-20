from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from pylab import *

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None

LEARNING_RATE = 0.04
NO_EPOCHS = 400
BATCH_SIZE = 100

def main(_):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])
  W = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10]))
  # Define linear classifier
  y = tf.sigmoid(tf.matmul(x, W) + b)

  
  y_ = tf.placeholder(tf.float32, [None, 10])

  # Define loss and optimizer //not working with reduce_sum, why ???
  loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
  train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)

  #accuracy tensors
  train_acc = []
  test_acc = []

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()



  # Train
  for i in range(NO_EPOCHS):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)

    _, acc = sess.run([train_step, accuracy], feed_dict={x: batch_xs, y_: batch_ys})
    train_acc.append(acc)
    test_acc.append(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                                  y_: mnist.test.labels}))

  print("train_acc: ",train_acc[-1],"  test_acc: ",test_acc[-1])

  plot ( arange(1,NO_EPOCHS+1),train_acc,color='g',label='train acc' )
  plot ( arange(1,NO_EPOCHS+1),test_acc,color='r',label='test acc' )
  xlabel('No epochs')
  ylabel('Accuracy')
  title('Linear - training and testing')
  legend(('train acc','test acc'))
  savefig("linear.png")

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

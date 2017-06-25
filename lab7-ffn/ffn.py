from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from pylab import *

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None

LEARNING_RATE = 0.4
NO_EPOCHS = 400
BATCH_SIZE = 100

def main(_):
  seed = 128
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # Create the model
  input_num_units = 784
  hidden_num_units = 300
  output_num_units = 10

  # Create the I/O
  x = tf.placeholder(tf.float32, [None, input_num_units])
  y_ = tf.placeholder(tf.float32, [None, output_num_units])

  # Create the weights
  W1 = tf.Variable(tf.random_normal([input_num_units, hidden_num_units], seed=seed))
  W2 = tf.Variable(tf.random_normal([hidden_num_units, output_num_units], seed=seed))

  b1 = tf.Variable(tf.random_normal([hidden_num_units], seed=seed))
  b2 = tf.Variable(tf.random_normal([output_num_units], seed=seed))

  # Create the layers
  hidden_layer = tf.add(tf.matmul(x, W1), b1)
  hidden_layer = tf.nn.sigmoid(hidden_layer)
  output_layer = tf.add(tf.matmul(hidden_layer, W2), b2)

  # Create the optimization
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=output_layer))
  train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)

  #Settings
  train_acc = []
  test_acc = []
  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()


    # Train
  for i in range(NO_EPOCHS):
    correct_prediction = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y_, 1))
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
  title('FFN - training and testing')
  legend(('train acc','test acc'))
  savefig("ffn.png")


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

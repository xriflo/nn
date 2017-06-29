#Source code with the blog post at http://monik.in/a-noobs-guide-to-implementing-rnn-lstm-using-tensorflow/
import numpy as np
import random
from random import shuffle
import tensorflow as tf
from pylab import *

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# from tensorflow.models.rnn import rnn_cell
# from tensorflow.models.rnn import rnn

NUM_EXAMPLES = 750

train_input = ['{0:010b}'.format(i) for i in range(2**10)]
shuffle(train_input)
train_input = [map(int,i) for i in train_input]
ti  = []
for i in train_input:
    temp_list = []
    for j in i:
            temp_list.append([j])
    ti.append(np.array(temp_list))
train_input = ti

train_output = []
for i in train_input:
    count = 0
    for j in i:
        if j[0] == 1:
            count+=1
    temp_list = ([0]*11)
    temp_list[count]=1
    train_output.append(temp_list)

test_input = train_input[NUM_EXAMPLES:]
test_output = train_output[NUM_EXAMPLES:]
train_input = train_input[:NUM_EXAMPLES]
train_output = train_output[:NUM_EXAMPLES]

print train_input[0]
print train_output[0]

print "test and training data loaded"


data = tf.placeholder(tf.float32, [None, 10,1]) #Number of examples, number of input, dimension of each input
target = tf.placeholder(tf.float32, [None, 11])
num_hidden = 24

cell = tf.nn.rnn_cell.LSTMCell(num_hidden,state_is_tuple=True)
val, _ = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)
val = tf.transpose(val, [1, 0, 2])
last = tf.gather(val, int(val.get_shape()[0]) - 1)
weight = tf.Variable(tf.truncated_normal([num_hidden, int(target.get_shape()[1])]))
bias = tf.Variable(tf.constant(0.1, shape=[target.get_shape()[1]]))
prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
cross_entropy = -tf.reduce_sum(target * tf.log(tf.clip_by_value(prediction,1e-10,1.0)))
optimizer = tf.train.AdamOptimizer()
minimize = optimizer.minimize(cross_entropy)
mistakes = tf.equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
accuracy = tf.reduce_mean(tf.cast(mistakes, tf.float32))


sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

batch_size = 10
no_of_batches = int(len(train_input)) / batch_size
epoch = 120
train_acc = []
test_acc = []
for i in range(epoch):
    ptr = 0
    for j in range(no_of_batches):
        inp, out = train_input[ptr:ptr+batch_size], train_output[ptr:ptr+batch_size]
        ptr+=batch_size
        sess.run(minimize,{data: inp, target: out})
    acc_tr = sess.run(accuracy, {data: train_input, target: train_output})
    acc_ts = sess.run(accuracy, {data: test_input, target: test_output})
    train_acc.append(acc_tr)
    test_acc.append(acc_ts)
    print("Epoch ", i, ": ", acc_ts)


plot ( arange(1,epoch+1),train_acc,color='g',label='train acc' )
plot ( arange(1,epoch+1),test_acc,color='r',label='test acc' )
xlabel('No epochs')
ylabel('Accuracy')
title('RNN - training and testing')
legend(('train acc','test acc'))
savefig("rnn.png")

sess.close()
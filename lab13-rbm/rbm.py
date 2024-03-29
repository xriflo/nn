from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
from PIL import Image
from util import tile_raster_images
import matplotlib.pyplot as plt

alpha = 1.0
batchsize = 100

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images,\
    mnist.test.labels

X = tf.placeholder("float", [None, 784])

rbm_w = tf.placeholder("float", [784, 500])
rbm_vb = tf.placeholder("float", [784])
rbm_hb = tf.placeholder("float", [500])

h0 = tf.nn.sigmoid(tf.matmul(X, rbm_w) + rbm_hb)
v1 = tf.nn.sigmoid(tf.matmul(h0, tf.transpose(rbm_w)) + rbm_vb)
h1 = tf.nn.sigmoid(tf.matmul(v1, rbm_w) + rbm_hb)

w_positive_grad = tf.matmul(tf.transpose(X), h0)
w_negative_grad = tf.matmul(tf.transpose(v1), h1)

update_w = rbm_w + alpha * (w_positive_grad - w_negative_grad) / tf.to_float(tf.shape(X)[0])
update_vb = rbm_vb + alpha * tf.reduce_mean(X - v1, 0)
update_hb = rbm_hb + alpha * tf.reduce_mean(h0 - h1, 0)

h_sample = tf.nn.sigmoid(tf.matmul(X, rbm_w) + rbm_hb)
v_sample = tf.nn.sigmoid(tf.matmul(h_sample, tf.transpose(rbm_w)) + rbm_vb)

err = X - v_sample
err_sum = tf.reduce_mean(err * err)

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

n_w = np.zeros([784, 500], np.float32)
n_vb = np.zeros([784], np.float32)
n_hb = np.zeros([500], np.float32)
o_w = np.zeros([784, 500], np.float32)
o_vb = np.zeros([784], np.float32)
o_hb = np.zeros([500], np.float32)


train_errors = [sess.run(err_sum, feed_dict={X: trX, rbm_w: o_w, rbm_vb: o_vb, rbm_hb: o_hb})]
test_errors = [sess.run(err_sum, feed_dict={X: teX, rbm_w: o_w, rbm_vb: o_vb, rbm_hb: o_hb})]
t = 0
for start, end in zip(range(0, len(trX), batchsize), range(batchsize, len(trX), batchsize)):
    batch = trX[start:end]
    n_w = sess.run(update_w, feed_dict={
                   X: batch, rbm_w: o_w, rbm_vb: o_vb, rbm_hb: o_hb})
    n_vb = sess.run(update_vb, feed_dict={
                    X: batch, rbm_w: o_w, rbm_vb: o_vb, rbm_hb: o_hb})
    n_hb = sess.run(update_hb, feed_dict={
                    X: batch, rbm_w: o_w, rbm_vb: o_vb, rbm_hb: o_hb})
    o_w = n_w
    o_vb = n_vb
    o_hb = n_hb
    if start % 10000 == 0:
        tr_err = sess.run(
            err_sum, feed_dict={X: trX, rbm_w: n_w, rbm_vb: n_vb, rbm_hb: n_hb})
        te_err = sess.run(
            err_sum, feed_dict={X: teX, rbm_w: n_w, rbm_vb: n_vb, rbm_hb: n_hb})
        train_errors.append(tr_err)
        test_errors.append(te_err)
        image = Image.fromarray(
            tile_raster_images(
                X=n_w.T,
                img_shape=(28, 28),
                tile_shape=(25, 20),
                tile_spacing=(1, 1)
            )
        )
        image.save("rbm_%d.png" % (t))
        t = t + 1

plt.plot(train_errors, color='g')
plt.plot(test_errors, color='r')
plt.title('RBM - training and testing')
plt.ylabel('Loss')
plt.xlabel('No epochs')
plt.legend(['train', 'test'])
plt.show()
from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.callbacks import Callback
import numpy as np
import matplotlib.pyplot as plt


(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))


print(x_train.shape)
print(x_test.shape)

IMG_DIM = 784
ENCODING_DIM = 32


input_img = Input(shape=(IMG_DIM,))
encoded = Dense(ENCODING_DIM, activation='relu')(input_img)
decoded = Dense(IMG_DIM, activation='relu')(encoded)

autoencoder = Model(inputs=input_img, outputs=decoded)
encoder = Model(inputs=input_img, outputs=encoded)

encoded_input = Input(shape=(ENCODING_DIM,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(inputs=encoded_input, outputs=decoder_layer(encoded_input))


autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
history = autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=100,
                shuffle=True,
                validation_data=(x_test, x_test))


encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

plt.plot(history.history['loss'], color='g')
plt.plot(history.history['val_loss'], color='r')
plt.title('VAE - training and testing')
plt.ylabel('Loss')
plt.xlabel('No epochs')
plt.legend(['train', 'test'])
plt.show()
# Tudor Berariu, 2016

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from mnist_loader import load_mnist

EPOCHS_NO = 400
LEARNING_RATE = 0.004
REPORT_EVERY = 40

def evaluate(model, X, L):
    Y = model.output(X)
    C = Y.argmax(1)
    (N, K) = Y.shape

    accuracy = (np.sum(C == L) * 1.0) / N

    conf_matrix = np.zeros((K, K), dtype="int")
    for i in range(N):
        conf_matrix[L[i],C[i]] += 1

    return accuracy, conf_matrix

def plot_confusion_matrix(conf_matrix, figure_id, title):
    plt.figure(figure_id)
    (N,_) = conf_matrix.shape
    plt.imshow(conf_matrix, interpolation='nearest')
    plt.xticks(np.arange(0,N), map(str,range(N)))
    plt.yticks(np.arange(0,N), map(str,range(N)))
    plt.title(title)

data = load_mnist()

N_train = data["train_no"]
#N_train = 20000
X_train = data["train_imgs"].squeeze()[:N_train,:]
L_train = data["train_labels"][:N_train]
T_train = np.zeros((N_train, L_train.max() + 1))
T_train[np.arange(N_train), L_train] = 1

N_test = data["test_no"]
X_test = data["test_imgs"].squeeze()
L_test = data["test_labels"]
T_test = np.zeros((N_test, L_test.max() + 1))
T_test[np.arange(N_test), L_test] = 1


print(N_test)
print(X_test.shape)
print(L_test.shape)
print(T_test.shape)



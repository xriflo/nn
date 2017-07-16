import numpy as np
from layer import *
from mnist_loader import load_mnist
from pylab import *
# 1. Create dataset

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


# Create network

def evaluate(model, X, L):
    Y = network.forward(X)
    C = Y.argmax(1)
    (N, K) = Y.shape

    accuracy = (np.sum(C == L) * 1.0) / N

    conf_matrix = np.zeros((K, K), dtype="int")
    for i in range(N):
        conf_matrix[L[i],C[i]] += 1

    return accuracy, conf_matrix

NO_EPOCHS = 7
network = Network()
network.addLayer(Linear(784,300))
network.addLayer(Tanh())
network.addLayer(Linear(300,10))
network.addLayer(CrossEntropy())

train_acc = []
test_acc = []

ep = 1
while ep <= NO_EPOCHS:
    network.forward(X_train)
    network.backward(X_train, T_train)
    network.updateParams(0.04)
    acc1, conf_matrix1 = evaluate(network, X_train, L_train)
    train_acc.append(acc1)
    acc2, conf_matrix2 = evaluate(network, X_test, L_test)
    test_acc.append(acc2)
    print '[Ep=%d] tr=%.2f ts=%.2f'%(ep, acc1, acc2)
    ep = ep + 1

plot ( arange(1,NO_EPOCHS+1),train_acc,color='g',label='train acc' )
plot ( arange(1,NO_EPOCHS+1),test_acc,color='r',label='test acc' )
xlabel('No epochs')
ylabel('Accuracy')
title('FFN - training and testing')
legend(('train acc','test acc'))
savefig("ffn.png")

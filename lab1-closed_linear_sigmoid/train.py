# Tudor Berariu, 2016

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from mnist_loader import load_mnist

from linear_classifier import LinearClassifier
from sigmoid_classifier import SigmoidClassifier

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

# ------------------------------------------------------------------------------
# ------ Closed form solution

cf_model = LinearClassifier()
cf_model.closed_form(X_train, T_train)

acc, conf = evaluate(cf_model, X_test, L_test)

print("[Closed Form] Accuracy on test set: %f" % acc)
print(conf)
plot_confusion_matrix(conf, 1, "Closed form")

acc1 = np.ones(EPOCHS_NO) * acc

print("-------------------")

# ------------------------------------------------------------------------------
# ------ Gradient optimization of linear model

grad_model = LinearClassifier()

acc2 = np.zeros(EPOCHS_NO)

ep = 1
while ep <= EPOCHS_NO:
    grad_model.update_params(X_train, T_train, LEARNING_RATE)
    acc, conf = evaluate(grad_model, X_test, L_test)

    acc2[ep-1] = acc

    if ep % REPORT_EVERY == 0:
        print("[Linear-grad] Epoch %4d; Accuracy on test set: %f" % (ep, acc))

    ep = ep + 1

print(conf)
plot_confusion_matrix(conf, 2, "Linear model - gradient")

print("-------------------")

# ------------------------------------------------------------------------------
# ------ Non-linear model

sig_model = SigmoidClassifier()

ep = 1
acc3 = np.zeros(EPOCHS_NO)

while ep <= EPOCHS_NO:
    sig_model.update_params(X_train, T_train, LEARNING_RATE)
    acc, conf = evaluate(sig_model, X_test, L_test)

    acc3[ep-1] = acc

    if ep % REPORT_EVERY == 0:
        print("[Linear-grad] Epoch %4d; Accuracy on test set: %f" % (ep, acc))

    ep = ep + 1

print(conf)
plot_confusion_matrix(conf, 3, "Sigmoid model")

print("-------------------")

plt.figure(4)

plt.plot(np.arange(1, EPOCHS_NO+1), acc1, label="Closed form")
plt.plot(np.arange(1, EPOCHS_NO+1), acc2, label="Linear model")
plt.plot(np.arange(1, EPOCHS_NO+1), acc3, label="Non-linear model")
plt.legend(loc="lower right")
plt.show()

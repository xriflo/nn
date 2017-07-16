# Tudor Berariu, 2016

import numpy as np
import math

def sigmoid(arr):
    return 1.0 / (1.0 + np.exp(-arr))

class SigmoidClassifier:

    def __init__(self):
        self.params = None

    def output(self, X):
        assert self.params is not None, "No parameters"

        ## TODO: Replace this code with a correct implementation
        (N, D) = X.shape
        (_, K) = self.params.shape

        Y = np.zeros((N, K))
        X_tilda = np.ones((N, D+1))
        X_tilda[:,0:D] = X

        Y = sigmoid(np.dot(X_tilda, self.params))
        ## ----

        return Y

    def update_params(self, X, T, lr):
        (N, D) = X.shape
        (_N, K) = T.shape
        assert _N == N, "X and T should have the same number of rows"

        if self.params is None:
            self.params = np.random.randn(D + 1, K) / 100

        X_tilda = np.ones((N, D+1))
        X_tilda[:,0:D] = X
        Y = self.output(X)
        dw = np.dot(np.transpose(((Y-T)*Y*(1-Y))), X_tilda)
        dw = dw/N
        self.params = self.params - lr*np.transpose(dw)
        ## TODO: Compute the gradient and update the parameters

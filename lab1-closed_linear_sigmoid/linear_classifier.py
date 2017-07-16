# Tudor Berariu, 2016

import numpy as np

class LinearClassifier:

    def __init__(self):
        self.params = None

    def closed_form(self, X, T):
        (N, D) = X.shape
        (_N, K) = T.shape
        assert _N == N, "X and T should have the same number of rows"

        ## TODO: Compute the values of the parameters
        self.params = np.zeros((D+1, K))
        X_tilda = np.ones((N, D+1))
        X_tilda[:,0:D] = X

        pinvf = np.linalg.pinv(np.dot(np.transpose(X_tilda), X_tilda))
        self.params = np.dot(pinvf, np.transpose(X_tilda))
        self.params = np.dot(self.params, T)

    def output(self, X):
        assert self.params is not None, "No parameters"

        ## TODO: Replace this code with a correct implementation
        (N, D) = X.shape
        (_, K) = self.params.shape

        Y = np.zeros((N, K))
        X_tilda = np.ones((N, D+1))
        X_tilda[:,0:D] = X
        
        Y = np.dot(X_tilda, self.params)

        return Y

    def update_params(self, X, T, lr):
        (N, D) = X.shape
        (_N, K) = T.shape
        assert _N == N, "X and T should have the same number of rows"

        if self.params is None:
            self.params = np.random.randn(D + 1, K) / 100

        X_tilda = np.ones((N, D+1))
        X_tilda[:,0:D] = X

        dw = np.subtract(np.dot(X_tilda, self.params), T)
        dw = np.dot(np.transpose(dw), X_tilda)
        dw = dw/N
        self.params = self.params - lr*np.transpose(dw)
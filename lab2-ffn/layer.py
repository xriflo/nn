# Florentina Bratiloveanu, 2016

import numpy as np
import sys
thismodule = sys.modules[__name__]

class Layer:
	def __init__(self):
		pass

	def forward(self, X):
		print "forward"

	def backward(self, X, dy):
		print "backward"

	def updateParams(self, eta):
		pass

	def printParams(self):
		print self.params

	def getLayerName(self):
		pass

class Tanh(Layer):
	def forward(self, X):
		var = np.tanh(X)
		return var

	def backward(self, X, dout):
		deriv = 1 - np.power(np.tanh(X), 2)
		var = np.multiply(dout, deriv)
		return var

	def getLayerName(self):
		return "Tanh"

class Logistic(Layer):
	def forward(self, X):
		var = 1/(1+np.exp(-X))
		return var

	def backward(self, X, dout):
		deriv = self.forward(X)*(1-self.forward(X))
		var = np.multiply(dout, deriv)
		return var

	def getLayerName(self):
		return "Logistic"

class Linear(Layer):
	def __init__(self, inputDimension, outputDimension):
		self.params = None
		self.inputDimension = inputDimension
		self.outputDimension = outputDimension
		self.weight = np.random.randn(inputDimension, outputDimension) / 100
		self.bias = np.random.randn(1, outputDimension) / 100
		self.N = None

	def forward(self, X):
		var = np.dot(X,self.weight)+self.bias
		return var

	def backward(self, X, dout):
		self.X = X
		self.dout = dout
		(N, K) = self.X.shape
		dX = np.dot(dout, np.transpose(self.weight))
		dw = np.dot(np.transpose(X), dout)/N
		db = np.sum(dout, axis=0)/N
		return dX, dw, db

	def getLayerName(self):
		return "Linear"

	def updateParams(self, eta):
		dX, dw, db = self.backward(self.X, self.dout)
		#print("self.weight before", self.weight[0,0:2])
		#print("dw: ", dw[0,0:2])
		self.weight = self.weight - eta*dw
		#print("self.weight after", self.weight[0,0:2])
		self.bias = self.bias - eta*db


class CrossEntropy(Layer):
	def __init__(self):
		pass

	def forward(self, X):
		e_X = np.exp(X)
		var = e_X/np.sum(e_X)
		return var

	def backward(self, X, t):
		e_X = np.exp(X)
		y = e_X/np.sum(e_X)
		var = y-t
		return var

	def getLayerName(self):
		return "SoftMax"

class OptimizationFunction:
	def __init__(self, net):
		self.net = net

	def train(self):
		pass

	def evaluate(self):
		pass

class SGD(OptimizationFunction):
	def __init__(self, net, dataset):
		pass
	def train(self):
		pass

class Network(Layer):
	def __init__(self):
		self.layers=[]
		self.y = []
		self.delta = []
	def zeroGradients(self):
		pass

	def addLayer(self, layer):
		self.layers.append(layer)

	def printNetwork(self):
		print "Network"

	def forward(self, X):
		self.y = [X]
		for i in range(0,len(self.layers)):
			self.y.append(self.layers[i].forward(self.y[i]))
		return self.y[-1]


	def backward(self, X, T):
		for i in range(0,len(self.layers)+1):
			self.delta.append(0)
		self.delta[-1] = T

		for i in range(len(self.layers)-1,-1,-1):
			if self.layers[i].getLayerName()=="Linear":
				dout, _, _ = self.layers[i].backward(self.y[i], self.delta[i+1])
			else:
				dout = self.layers[i].backward(self.y[i], self.delta[i+1])
			self.delta[i] = dout

	def updateParams(self, eta):
		for layer in self.layers:
			layer.updateParams(eta)

def fshape(x):
	return x.shape
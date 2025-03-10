#Feedforward Neural Network with Backpropagation

import numpy as np
# X = (hours
X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
y = np.array (([92], [86], [89]), dtype=float)
X = X/np.amax (X, axis=0)
# maximum of X array
y = y/100
# max test score is 100
class Neural_Network(object):
    def __init__(self):
        self.inputSize = 2
        self.outputSize = 1
        self.hiddenSize = 3
        self.W1 = np.random.randn(self.inputSize,self.hiddenSize)
        self.W2 = np.random.randn(self.hiddenSize,self.outputSize)
    def forward (self, X):
        self.z = np.dot(X, self.W1)
        self.z2 = self.sigmoid(self.z)
        self.z3 = np.dot(self.z2, self.W2)
        o = self.sigmoid(self.z3)
        return o
    def sigmoid(self, s): 
        return 1/(1+np.exp(-s))
    def sigmoidPrime (self, s):
        return s * (1-s)
    def backward(self, X, y, o):
        self.o_error = y-o
        self.o_delta = self.o_error*self.sigmoidPrime(o)
        self.z2_error = self.o_delta.dot(self.W2.T)
        self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2)
        self.W1 += X.T.dot(self.z2_delta)
    def train(self, X, y):
        o = self.forward(X)
        self.backward(X,y,o)
NN = Neural_Network()
for i in range(1000):
    print("\nInput: \n"+str(X))
    print("\nActual Output: \n"+str(y))
    print("\nPredicted Output: \n"+str(NN.forward(X)))
    print("\nLoss: \n"+str(np.mean(np.square(y-NN.forward(X)))))   
    NN.train(X, y)        
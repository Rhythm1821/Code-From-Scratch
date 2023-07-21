import numpy as np

class NeuralNetwork:
    def __init__(self,X,y):
        self.X = X
        self.y = y
        self.weights = np.random.randn(X.shape[1],1)
        self.bias = np.random.randn(X.shape[0])
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))
    def forward(self):
        z = np.dot(self.X,self.weights)+self.bias
        return self.sigmoid(z)
    def loss(self):
        y_pred = self.forward()
        loss = -(self.y*np.log(y_pred) + (1 - self.y) * np.log(1 - y_pred))
        return loss.mean()
    def backward(self):
        pass
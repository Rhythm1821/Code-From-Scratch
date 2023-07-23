import numpy as np

class NeuralNetwork:
    def __init__(self,input_size,hidden_size,output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size= output_size

        # Initialize weights and biases for hidden layer
        self.W1 = np.random.randn(self.input_size,self.output_size)
        self.b1 = np.zeros((1,self.hidden_size))

        # Initilize weights and biases for the output layer
        self.W2 = np.random.randn(self.hidden_size,self.output_size)
        self.b2 = np.zeros((1,self.output_size))

    def forward(self,X):
        # forward propagation
        self.z1 = np.dot(X,self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1,self.W2) + self.b2
        self.a2 = self.softmax(self.z2)
        return self.a2
    
    def backward(self,X,y,learning_rate):
        m = X.shape

        # Output layer gradients
        dZ2 = self.a2 - y
        dW2 = np.dot(self.a1.T,dZ2) / m
        db2 = np.sum(dZ2,axis=0,keepdims=True) / m

        # Hidden layer gradients
        dZ1 = np.dot(dZ2,self.W2.T) * self.sigmoid_derivative(self.z1)
        dW1 = np.dot(X.T,dZ1) / m
        db1 = np.sum(dZ1,axis=0,keepdims=True) / m

        # Update weights and biases
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    def sigmoid(self,z):
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(self,z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))
    
    def softmax(self,z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def train(self,X,y,epochs,learning_rate):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, learning_rate)
    
    def predict(self,X):
        return np.argmax(self.forward(self.forward(X), axis=1))
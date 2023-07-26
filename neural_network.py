import numpy as np

class NeuralNetwork:
    def __init__(self,input_size,hidden_size,output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size= output_size

        # Initialize weights and biases for hidden layer
        self.W1 = np.random.randn(self.input_size,self.hidden_size)
        self.b1 = np.zeros((1,self.hidden_size))

        # Initilize weights and biases for the output layer
        self.W2 = np.random.randn(self.hidden_size,self.output_size)
        self.b2 = np.zeros((1,self.output_size))

    def forward(self,X):
        # forward propagation
        self.hidden_output = self.sigmoid(np.dot(X,self.W1) + self.b1)
        self.output = self.sigmoid(np.dot(self.hidden_output,self.W2) + self.b2)

        return  self.output
    
    def backward(self,X,y,lr):
        d_output = (y - self.output) * self.sigmoid_derivative(self.hidden_output)
        d_W2 = np.dot(self.hidden_output.T,d_output)
        d_b2 = np.sum(d_output,axis=1,keepdims=True)
        
        d_hidden = np.dot(d_output,self.W2.T) * self.sigmoid_derivative(self.hidden_output)
        d_W1 = np.dot(X.T,d_hidden)
        d_b1 = np.sum(d_hidden,axis=1,keepdims=True)

        # Update weights and biases
        self.W2 += lr * d_W2
        self.b2 += lr * d_b2
        self.W1 += lr * d_W1
        self.b1 += lr * d_b1

    def sigmoid(self,z):
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(self,z):
        return z * (1 - z)
    
    def train(self,X,y,epochs,lr):
        for _ in range(epochs):
            output = self.forward(X)
            self.backward(X, y, lr)
            loss = np.mean((y - output)**2)
    
    def predict(self,X):
        return self.forward(X)
    
































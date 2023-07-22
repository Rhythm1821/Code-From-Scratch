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
        y_pred = self.forward()
        m = self.X.shape[0]

        # Compute gradient using the chain rule
        dL_dy_pred = -(self.y / y_pred) + ((1 - self.y) / (1 - y_pred))
        dy_pred_dz = y_pred * (1 - y_pred)
        dL_dz = dL_dy_pred * dy_pred_dz

        # Compute gradients with respect to weights and biases
        dL_dw = (1 / m) * np.dot(self.X.T, dL_dz)
        dL_db = (1 / m) * np.sum(dL_dz, axis=1, keepdims=True)
        
        return dL_dw, dL_db
    
    def train(self,epochs,lr):
        for epoch in range(epochs):
            y_pred = self.forward()
            loss = self.loss()
            dL_dw, dL_db = self.backward()
            print(self.weights)
            print(dL_dw)
            # print()
            self.weights -= lr * dL_dw
            self.bias -= lr * dL_db

            #  print the loss for monitoring training progress
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")        
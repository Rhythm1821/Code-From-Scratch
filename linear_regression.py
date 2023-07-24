import numpy as np

class LinearRegression:
    def __init__(self,lr=0.01,iterations=1000):
        self.lr = lr
        self.iterations = iterations
        self.weights = None
        self.bias = None
    def fit(self,X,y):
        num_samples, num_features = X.shape

        # Initialize the weights and bias to zeros
        self.weights = np.zeros(num_features)
        self.bias = 0

        # Gradient descent
        for _ in range(self.iterations):
            # Predict the output using the weights and bias 
            y_pred = self.predict(X)

            # Calculate the gradients
            dw = (1 / num_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / num_samples) * np.sum(y_pred - y)

            # Update the weights and biases
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
    def predict(self):
        pass
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.datasets import make_regression

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
            
    def predict(self,X):
        return np.dot(X,self.weights) + self.bias

X,y = make_regression(n_samples=100,
                        n_features=2,
                        noise=1,random_state=42)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

lr = LinearRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)

mse = mean_squared_error(y_pred=y_pred,y_true=y_test)
mae = mean_absolute_error(y_pred=y_pred,y_true=y_test)
print(mse,mae)
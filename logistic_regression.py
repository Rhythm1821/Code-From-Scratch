import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# from sklearn.linear_model import LogisticRegression

class LogisticRegression:
    def __init__(self,lr=0.01,iterations=1000):
        self.lr = lr
        self.iterations = iterations
        self.weights = None
        self.bias = None
    def fit(self,X,y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        # Gradient descent
        for _ in range(self.iterations):
            linear_model = np.dot(X,self.weights) + self.bias
            y_pred = self._sigmoid(linear_model)

            dw = (1 / num_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / num_samples) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    def predict(self,x):
        linear_model = np.dot(x,self.weights) + self.bias
        y_pred = self._sigmoid(linear_model)
        y_pred_class = [1 if i > 0.5 else 0 for i in y_pred]
        return y_pred_class
    def _sigmoid(self,z):
        return 1 / (1 + np.exp(-z))





"""
Check if it works

X,y = make_classification(n_samples=1000,n_classes=2,n_features=7)

X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

lr = LogisticRegression()

lr.fit(X_train,y_train)

y_pred = lr.predict(X_test)

score = accuracy_score(y_test,y_pred)

print(score)
"""
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class SVM:
    def __init__(self,lr=0.001,lambda_param=0.01,n_iters=1000):
        self.lr = lr
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None
    def fit(self,X,y):
        y_ = np.where(y<=0,-1,1)
        num_samples,num_features = X.shape

        self.w = np.zeros(num_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i,self.w)-self.b) >=1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -=self.lr * (2 * self.lambda_param * self.w - np.dot(x_i,y_[idx]))
                    self.b -= self.lr * y_[idx]
    def predict(self,X):
        linear_output = np.dot(X,self.w) - self.b
        return np.sign(linear_output)


"""
X,y = make_blobs(n_samples=50,
           n_features=2,
           centers=2,
           cluster_std=1.05,
           random_state=40)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)


svm = SVM()

svm.fit(X_train,y_train)

y_pred = svm.predict(X_test)

score = accuracy_score(y_pred=y_pred,y_true=y_test)

print(score)
"""
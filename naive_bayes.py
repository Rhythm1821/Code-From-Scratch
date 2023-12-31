import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class NaiveBayes:
    def __init__(self):
        pass
    def fit(self,X,y):
        num_samples,num_features = X.shape
        self._classes = np.unique(y)
        num_classes = len(self._classes)

        # init mean, var, priors
        self._mean = np.zeros((num_classes,num_features),dtype=np.float64)
        self._var = np.zeros((num_classes,num_features),dtype=np.float64)
        self._priors = np.zeros(num_classes,dtype=np.float64)

        for c in self._classes:
            X_c = X[c==y]
            self._mean[c,:] = X_c.mean(axis=0)
            self._var[c,:] = X_c.var(axis=0)
            self._priors[c] = X_c.shape[0] / float(num_samples)
    def predict(self,X):
        y_pred = [self._predict(x) for x in X]
        return y_pred
    def _predict(self,x):
        posteriors = []

        for idx,c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            class_conditional = np.sum(np.log(self._pdf(idx,x)))
            posterior = prior + class_conditional
            posteriors.append(posterior)

        return self._classes[np.argmax(posteriors)]
    def _pdf(self,class_idx,x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(-(x - mean)**2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)

        return numerator / denominator


"""Evaluating the model
X,y = make_classification(n_samples=1000,
                    n_features=10,
                    n_classes=2,
                    random_state=123)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=123)

nb = NaiveBayes()

nb.fit(X_train,y_train)

y_pred = nb.predict(X_test)

accuracy = accuracy_score(y_pred=y_pred,
               y_true=y_test)

print(f'Accuracy for your model is {accuracy*100}%')
"""
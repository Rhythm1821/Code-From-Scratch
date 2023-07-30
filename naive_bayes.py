import numpy as np

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
            self._priors[c,:] = X_c.shape[0] / float(num_samples)
    def predict(self,X):
        pass
import numpy as np

class KNNClassifier:
    def __init__(self,k=5):
        self.k = k
    def fit(self,X,y):
        self.X_train = X
        self.y_train = y
    def euclidian_distance(self,x1,x2):
        return np.sqrt(np.sum((x1 - x2)** 2))
    def predict(self,X):
        y_pred = [self._predict_single(x) for x in X]
        return np.array(y_pred)
    def _predict_single(self,x):
        # Calculate distances between x and all examples in the training set
        distances = [self.euclidian_distance(x,x_train) for x_train in self.X_train]

        # Get the indices of the k-nearest neighbors
        k_nearest_labels = np.argsort(distances)[:self.k]

        # Return the most common label among the k-nearest neighbours
        most_common = np.bincount(k_nearest_labels).argmax()

        return most_common
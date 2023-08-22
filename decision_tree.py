import numpy as np

def entropy(y):
    hist = np.bincount(y)
    ps = hist/len(y)
    return np.sum([p * np.log2(p) for p in ps if p > 0])

class Node:
    def __init__(self,feature=None,threshold=None,left=None,right=None,*,value=None):
        self.feature = feature
        self.threshold = threshold
        self.right = right
        self.left = left
        self.value = value
    
    def is_leaf_node(self):
        return self.value is not None

class DecisionTree:
    def __init__(self,min_samples_split=2,max_depth=100,n_feats=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.root = None
    
    def fit(self,X,y):
        # grow tree
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats,X.shape[1])
        self.root=self._grow_tree(X,y)
    
    def _grow_tree(self,X,y,depth=0):
        num_samples,num_features = X.shape
        n_labels=len(np.unique(y))

        # Stopping criteria
        if (depth>self.max_depth
            or n_labels==1
            or num_samples<self.min_samples_split):
            leaf_value=self._most_common_labels(y)
            return Node(value=leaf_value)
        feat_idxs = np.random.choice(num_features,self.n_feats)

        # greedy search
        pass
    
    def predict(self,X):
        # traverse tree
        pass

    def _most_common_labels(self,y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common
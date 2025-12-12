import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from .base_model import BaseModel


class KNNModel(BaseModel):
    
    def __init__(self, n_neighbors=5, **kwargs):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.kwargs = kwargs
        self.model = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            **self.kwargs
        )
    
    def fit(self, X, y):
        X = self._validate_input(X)
        y = np.asarray(y)
        
        self.model.fit(X, y)
        self.is_fitted = True
        
        return self
    
    def predict(self, X):
        self._validate_fitted()
        X = self._validate_input(X)
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        self._validate_fitted()
        X = self._validate_input(X)
        
        return self.model.predict_proba(X)

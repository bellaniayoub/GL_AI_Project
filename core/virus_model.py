from .base_model import BaseModel
from .knn import KNNModel
from .decision_tree import DecisionTreeModel

class VirusDiag(BaseModel):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def fit(self, X, y):
        self.model.fit(X, y)
        self.is_fitted = True
        return self
        
    def predict(self, X):
        self._validate_fitted()
        return self.model.predict(X)
        
    def predict_proba(self, X):
        self._validate_fitted()
        return self.model.predict_proba(X)
        
    def get_model(self):
        return self.model

class VirusModelBuilder:
    def __init__(self):
        self.model_type = None
        self.start_hyperparameters = {}
        
    def set_model_type(self, model_type):
        if model_type not in ['knn', 'decision_tree']:
            raise ValueError("Model type must be 'knn' or 'decision_tree'")
        self.model_type = model_type
        return self
        
    def set_hyperparameters(self, **kwargs):
        self.start_hyperparameters = kwargs
        return self
        
    def build(self):
        if self.model_type == 'knn':
            model = KNNModel(**self.start_hyperparameters)
        elif self.model_type == 'decision_tree':
            model = DecisionTreeModel(**self.start_hyperparameters)
        else:
            raise ValueError("Model type not set or invalid")
            
        return VirusDiag(model)

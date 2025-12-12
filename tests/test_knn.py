import sys
import os
import pytest
import numpy as np
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.knn import KNNModel
from pipeline.trainer import Trainer

def test_knn_initialization():
    """Test that KNNModel can be initialized with default and custom parameters."""
    model = KNNModel()
    assert model.n_neighbors == 5
    
    model_custom = KNNModel(n_neighbors=3)
    assert model_custom.n_neighbors == 3

def test_knn_training_prediction():
    """Test KNNModel training and prediction with dummy data."""
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.array([0, 0, 1, 1])
    
    model = KNNModel(n_neighbors=1)
    model.fit(X, y)
    
    assert model.is_fitted
    
    predictions = model.predict(X)
    assert len(predictions) == 4
    assert np.array_equal(predictions, y)

def test_knn_trainer_integration(tmp_path):
    """Test that KNNModel works with the Trainer class."""
    models_dir = tmp_path / "Models"
    trainer = Trainer(models_dir=str(models_dir))
    
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.array([0, 0, 1, 1])
    model = KNNModel(n_neighbors=1)
    
    # Test training and saving
    trained_model, model_path = trainer.train(model, X, y, model_name="test_knn")
    
    assert trained_model.is_fitted
    assert os.path.exists(model_path)
    
    # Test loading
    loaded_model = trainer.load_model(model_path)
    assert loaded_model is not None
    assert isinstance(loaded_model, KNNModel)
    assert loaded_model.n_neighbors == 1

if __name__ == "__main__":
    # Allow running directly
    try:
        test_knn_initialization()
        print("test_knn_initialization passed")
        
        test_knn_training_prediction()
        print("test_knn_training_prediction passed")
        
        # Mock tmp_path for script execution
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_knn_trainer_integration(Path(tmp_dir))
        print("test_knn_trainer_integration passed")
        
        print("\nAll KNN tests passed!")
    except Exception as e:
        print(f"\nTest failed: {e}")
        sys.exit(1)

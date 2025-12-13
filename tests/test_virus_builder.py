import sys
import os
import pytest
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.virus_model import VirusModelBuilder, VirusDiag
from core.knn import KNNModel
from core.decision_tree import DecisionTreeModel
from pipeline.trainer import Trainer

def test_virus_builder_knn():
    """Test that VirusModelBuilder can build a KNN model."""
    builder = VirusModelBuilder()
    builder.set_model_type('knn')
    builder.set_hyperparameters(n_neighbors=3)
    
    virus_model = builder.build()
    
    assert isinstance(virus_model, VirusDiag)
    assert isinstance(virus_model.get_model(), KNNModel)
    assert virus_model.get_model().n_neighbors == 3

def test_virus_builder_decision_tree():
    """Test that VirusModelBuilder can build a Decision Tree model."""
    builder = VirusModelBuilder()
    builder.set_model_type('decision_tree')
    builder.set_hyperparameters(max_depth=5)
    
    virus_model = builder.build()
    
    assert isinstance(virus_model, VirusDiag)
    assert isinstance(virus_model.get_model(), DecisionTreeModel)
    assert virus_model.get_model().max_depth == 5

def test_virus_diag_training_prediction():
    """Test VirusDiag training and prediction."""
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.array([0, 0, 1, 1])
    
    builder = VirusModelBuilder()
    builder.set_model_type('knn').set_hyperparameters(n_neighbors=1)
    
    virus_model = builder.build()
    virus_model.fit(X, y)
    
    assert virus_model.is_fitted
    
    predictions = virus_model.predict(X)
    assert len(predictions) == 4
    assert np.array_equal(predictions, y)

def test_trainer_integration(tmp_path):
    """Test that VirusModelBuilder works with the Trainer class."""
    models_dir = tmp_path / "Models"
    trainer = Trainer(models_dir=str(models_dir))
    
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.array([0, 0, 1, 1])
    
    builder = VirusModelBuilder()
    builder.set_model_type('knn').set_hyperparameters(n_neighbors=1)
    
    # Test training and saving
    trained_model, model_path = trainer.train_with_builder(builder, X, y, model_name="test_virus")
    
    assert isinstance(trained_model, VirusDiag)
    assert trained_model.is_fitted
    assert os.path.exists(model_path)
    
    # Test loading - note that load_model returns the pickle object which is the VirusDiag instance
    loaded_model = trainer.load_model(model_path)
    assert loaded_model is not None
    assert isinstance(loaded_model, VirusDiag)
    assert isinstance(loaded_model.get_model(), KNNModel)

if __name__ == "__main__":
    # Allow running directly
    try:
        test_virus_builder_knn()
        print("test_virus_builder_knn passed")
        
        test_virus_builder_decision_tree()
        print("test_virus_builder_decision_tree passed")
        
        test_virus_diag_training_prediction()
        print("test_virus_diag_training_prediction passed")
        
        # Mock tmp_path for script execution
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_trainer_integration(Path(tmp_dir))
        print("test_trainer_integration passed")
        
        print("\nAll VirusModelBuilder tests passed!")
    except Exception as e:
        print(f"\nTest failed: {e}")
        # Print full traceback for easier debugging
        import traceback
        traceback.print_exc()
        sys.exit(1)

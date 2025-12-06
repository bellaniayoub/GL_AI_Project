import sys
import os
import pytest
# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.trainer import Trainer
from core.dataset import RespiratoryInfectionDataset

def test_imports():
    """Test that main modules can be imported."""
    assert Trainer is not None
    assert RespiratoryInfectionDataset is not None

def test_trainer_initialization(tmp_path):
    """Test that Trainer initializes correctly."""
    models_dir = tmp_path / "Models"
    trainer = Trainer(models_dir=str(models_dir))
    assert os.path.exists(models_dir)
    assert trainer.models_dir == str(models_dir)

if __name__ == "__main__":
    test_imports()
    # We can't easily test test_trainer_initialization here because it needs tmp_path fixture
    print("Imports successful!")

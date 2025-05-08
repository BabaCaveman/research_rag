# This file makes the models directory a Python package
# This allows imports like: from models.model_loader import ModelLoader

from pathlib import Path

# Define the path to the models directory
MODELS_DIR = Path(__file__).parent.absolute()

# Export the ModelLoader class
__all__ = ['ModelLoader']
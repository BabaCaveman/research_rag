"""
Utility script to verify model loading
"""
import os
import sys
from backend.core.config import Config
from backend.models.model_loader import load_model_components

def verify_model_loading():
    """
    Test that model components can be loaded correctly
    """
    config = Config()
    
    print("==== Model Loading Verification ====")
    print(f"BASE_DIR: {config.BASE_DIR}")
    print(f"MODEL_DIR: {config.MODEL_DIR}")
    print("\nChecking model directory...")
    
    if os.path.exists(config.MODEL_DIR):
        print(f"✓ Model directory exists at: {config.MODEL_DIR}")
        
        # List files in the directory
        files = os.listdir(config.MODEL_DIR)
        print(f"\nFiles in model directory ({len(files)}):")
        for file in files:
            file_path = os.path.join(config.MODEL_DIR, file)
            file_size = os.path.getsize(file_path) / 1024
            print(f"  - {file} ({file_size:.2f} KB)")
        
        # Check expected model files
        print("\nChecking expected model files:")
        for component, filename in config.MODEL_FILES.items():
            file_path = os.path.join(config.MODEL_DIR, filename)
            if os.path.exists(file_path):
                print(f"✓ {component}: {filename} exists")
            else:
                print(f"✗ {component}: {filename} NOT FOUND")
        
        # Try to load all components
        print("\nAttempting to load all model components...")
        try:
            components = load_model_components()
            print(f"✓ Successfully loaded {len(components)} model components:")
            for component_name, component in components.items():
                print(f"  - {component_name}: {type(component).__name__}")
            print("\nModel verification completed successfully!")
            return True
        except Exception as e:
            print(f"✗ Error loading model components: {str(e)}")
            return False
    else:
        print(f"✗ Model directory not found at: {config.MODEL_DIR}")
        return False

if __name__ == "__main__":
    success = verify_model_loading()
    sys.exit(0 if success else 1)
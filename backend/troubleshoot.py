"""
Troubleshooting utility for the crop reflectance analysis application
"""
import os
import sys
import pandas as pd
import numpy as np
import joblib
import argparse
from backend.data.validation import validate_csv
from backend.data.preprocessing import preprocess_data, apply_savitzky_golay, normalize_data, apply_pca
from backend.models.model_loader import load_model, get_model_metadata
from backend.core.config import Config

# Load configuration
config = Config()

def check_environment():
    """Check environment setup"""
    print("Checking environment...")
    
    # Check Python version
    print(f"Python version: {sys.version}")
    
    # Check critical directories
    dirs_to_check = [
        config.TEMP_DIR,
        config.DATA_DIR,
        config.MODEL_DIR
    ]
    
    for dir_path in dirs_to_check:
        if os.path.exists(dir_path):
            print(f"✓ Directory exists: {dir_path}")
        else:
            print(f"✗ Directory missing: {dir_path}")
            try:
                os.makedirs(dir_path, exist_ok=True)
                print(f"  Created directory: {dir_path}")
            except Exception as e:
                print(f"  Failed to create directory: {e}")
    
    # Check model file
    if os.path.exists(config.MODEL_DIR):
        print(f"✓ Model file exists: {config.MODEL_DIR}")
        # Check if model loads correctly
        try:
            model = load_model()
            print(f"✓ Model loaded successfully: {type(model).__name__}")
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
    else:
        print(f"✗ Model file missing: {config.MODEL_DIR}")
    
    print("\nEnvironment check complete.")

def test_pipeline(csv_path):
    """Test the full data processing pipeline with a sample CSV"""
    print(f"\nTesting pipeline with CSV: {csv_path}")
    
    # Check if file exists
    if not os.path.exists(csv_path):
        print(f"✗ CSV file not found: {csv_path}")
        return
    
    try:
        # Step 1: Validate CSV
        print("Step 1: Validating CSV...")
        validation_result = validate_csv(csv_path)
        if validation_result["valid"]:
            print("✓ CSV validation passed")
        else:
            print(f"✗ CSV validation failed: {validation_result['error']}")
            return
        
        # Step 2: Load and preprocess data
        print("Step 2: Preprocessing data...")
        raw_data = pd.read_csv(csv_path)
        print(f"  - Raw data shape: {raw_data.shape}")
        
        preprocessed_data = preprocess_data(raw_data)
        print(f"  - Preprocessed data shape: {preprocessed_data.shape}")
        
        # Step 3: Apply Savitzky-Golay filter
        print("Step 3: Applying Savitzky-Golay filter...")
        filtered_data = apply_savitzky_golay(preprocessed_data)
        print(f"  - Filtered data shape: {filtered_data.shape}")
        
        # Step 4: Normalize data
        print("Step 4: Normalizing data...")
        normalized_data = normalize_data(filtered_data)
        print(f"  - Normalized data shape: {normalized_data.shape}")
        
        # Step 5: Apply PCA
        print("Step 5: Applying PCA...")
        pca_data = apply_pca(normalized_data)
        print(f"  - PCA data shape: {pca_data.shape}")
        
        # Step 6: Load model
        print("Step 6: Loading model...")
        model = load_model()
        print(f"  - Model type: {type(model).__name__}")
        
        # Step 7: Make predictions
        print("Step 7: Making predictions...")
        predictions = model.predict(pca_data)
        print(f"  - Predictions shape: {predictions.shape}")
        print(f"  - First few predictions: {predictions[:5]}")
        
        print("\n✓ Pipeline test completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Pipeline test failed: {str(e)}")

def main():
    """Main function for troubleshooting tool"""
    parser = argparse.ArgumentParser(description="Troubleshooting tool for crop reflectance analysis")
    parser.add_argument("--check-env", action="store_true", help="Check environment setup")
    parser.add_argument("--test-pipeline", type=str, help="Test pipeline with a sample CSV file")
    
    args = parser.parse_args()
    
    if args.check_env:
        check_environment()
    
    if args.test_pipeline:
        test_pipeline(args.test_pipeline)
    
    if not args.check_env and not args.test_pipeline:
        print("No action specified. Use --help for available options.")

if __name__ == "__main__":
    main()
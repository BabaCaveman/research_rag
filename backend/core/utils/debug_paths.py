"""
Simple script to debug paths and verify model files exist
"""
import os
import sys

def debug_paths():
    """Print all directory and file information for debugging"""
    # Get the absolute path of the current script
    current_script = os.path.abspath(__file__)
    print(f"Current script: {current_script}")
    
    # Get the directory containing the script
    script_dir = os.path.dirname(current_script)
    print(f"Script directory: {script_dir}")
    
    # Assume project structure: project_root/backend/models/[files]
    # Navigate up to find the models directory
    project_root = os.path.dirname(os.path.dirname(script_dir))
    print(f"Project root: {project_root}")
    
    # Check for models directory options
    models_path_option1 = os.path.join(project_root, 'models')
    models_path_option2 = os.path.join(project_root, 'backend', 'models')
    
    print(f"\nChecking models directory options:")
    print(f"Option 1: {models_path_option1}")
    print(f"  Exists: {os.path.exists(models_path_option1)}")
    
    print(f"Option 2: {models_path_option2}")
    print(f"  Exists: {os.path.exists(models_path_option2)}")
    
    # Find the correct models directory
    models_path = None
    if os.path.exists(models_path_option2):
        models_path = models_path_option2
    elif os.path.exists(models_path_option1):
        models_path = models_path_option1
    
    if models_path:
        print(f"\nFound models directory at: {models_path}")
        
        # List files in the models directory
        files = os.listdir(models_path)
        print(f"\nFiles in models directory ({len(files)}):")
        for file in files:
            file_path = os.path.join(models_path, file)
            file_size = os.path.getsize(file_path) / 1024
            print(f"  - {file} ({file_size:.2f} KB)")
        
        # Check for specific model files
        model_files = [
            'spectral_Moisture_rf_model.pkl',
            'scaler_Moisture.pkl',
            'pca_Moisture.pkl',
            'model_Moisture_metadata.pkl'
        ]
        
        print("\nChecking for specific model files:")
        for filename in model_files:
            file_path = os.path.join(models_path, filename)
            if os.path.exists(file_path):
                print(f"✓ {filename} exists")
            else:
                print(f"✗ {filename} NOT FOUND")
    else:
        print("\nCould not find models directory!")

if __name__ == "__main__":
    debug_paths()
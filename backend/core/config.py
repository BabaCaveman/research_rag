"""
Configuration settings for the research_rag application
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class for application settings"""
    
    def __init__(self):
        # API settings
        self.DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
        self.PORT = int(os.getenv('PORT', 5000))
        
        # Get the base directory - this needs to find the correct project root
        # The issue is that this is resolving incorrectly, causing the MODEL_DIR to be wrong
        current_file = os.path.abspath(__file__)
        core_dir = os.path.dirname(current_file)  # core directory
        backend_dir = os.path.dirname(core_dir)   # backend directory
        self.BASE_DIR = os.path.dirname(backend_dir)  # project root directory
        
        # Print the resolution of directories for debugging
        print(f"Current file: {current_file}")
        print(f"Core dir: {core_dir}")
        print(f"Backend dir: {backend_dir}")
        print(f"BASE_DIR: {self.BASE_DIR}")
        
        # Directory settings - explicitly set absolute paths
        self.TEMP_DIR = os.path.join(self.BASE_DIR, 'temp')
        self.DATA_DIR = os.path.join(self.BASE_DIR, 'backend', 'data')
        
        # Model settings - EXPLICITLY set the model directory to what we can see in your file explorer
        # Overriding with environment variable if set
        default_model_dir = os.path.join(self.BASE_DIR, 'backend', 'models')
        self.MODEL_DIR = os.getenv('MODEL_DIR', default_model_dir)
        
        print(f"MODEL_DIR resolved to: {self.MODEL_DIR}")
        
        # Model filenames - these must match exactly what's in your directory
        self.MODEL_FILES = {
            'rf_model': 'spectral_Moisture_rf_model.pkl',
            'scaler': 'scaler_Moisture.pkl',
            'pca': 'pca_Moisture.pkl',
            'metadata': 'model_Moisture_metadata.pkl'
        }
        
        # Preprocessing settings
        self.SAVGOL_WINDOW = int(os.getenv('SAVGOL_WINDOW', 15))
        self.SAVGOL_POLYORDER = int(os.getenv('SAVGOL_POLYORDER', 3))
        self.PCA_COMPONENTS = int(os.getenv('PCA_COMPONENTS', 10))
        
        # Create required directories
        os.makedirs(self.TEMP_DIR, exist_ok=True)
        os.makedirs(self.MODEL_DIR, exist_ok=True)
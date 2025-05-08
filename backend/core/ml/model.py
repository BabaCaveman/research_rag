# core/ml/model.py
"""
Model loading and prediction functionality.
"""

import joblib
import os
from flask import current_app
import numpy as np

def load_model():
    """Load the pre-trained model"""
    try:
        model_path = current_app.config['MODEL_PATH']
        if not os.path.exists(model_path):
            current_app.logger.error(f"Model file not found at {model_path}")
            return None
            
        model = joblib.load(model_path)
        return model
    except Exception as e:
        current_app.logger.error(f"Error loading model: {str(e)}")
        return None

def predict(model, processed_data):
    """Make predictions using the provided model"""
    # Make class predictions
    predictions = model.predict(processed_data)
    
    # Get prediction probabilities if model supports it
    probabilities = None
    if hasattr(model, 'predict_proba'):
        try:
            probabilities = model.predict_proba(processed_data)
        except:
            pass
    
    return predictions, probabilities
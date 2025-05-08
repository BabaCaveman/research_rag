# core/ml/preprocessing.py
"""
Data preprocessing functionality for spectral data.
"""

import numpy as np
import joblib
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from flask import current_app
import os

def smooth_data(data, window_length=11, polyorder=2):
    """Apply Savitzky-Golay filtering to smooth spectral data"""
    return savgol_filter(data, window_length=window_length, polyorder=polyorder)

def normalize_data(data, scaler=None):
    """
    Normalize spectral data using StandardScaler
    If scaler is provided, transform data using it, otherwise fit a new scaler
    """
    if scaler is None:
        scaler = StandardScaler()
        return scaler.fit_transform(data), scaler
    else:
        return scaler.transform(data), scaler

def apply_pca(data, pca=None, n_components=0.95):
    """
    Apply PCA to reduce dimensionality
    If PCA object is provided, transform data using it, otherwise fit a new PCA
    """
    if pca is None:
        pca = PCA(n_components=n_components)
        return pca.fit_transform(data), pca
    else:
        return pca.transform(data), pca

def preprocess_data(data):
    """Apply full preprocessing pipeline to spectral data"""
    try:
        # Check if preprocessing objects exist
        preproc_path = current_app.config['PREPROCESSING_PATH']
        
        if os.path.exists(preproc_path):
            # Load existing preprocessing objects
            scaler, pca = joblib.load(preproc_path)
            
            # Apply preprocessing pipeline with saved objects
            # Smooth data using Savitzky-Golay filter
            smoothed_data = np.apply_along_axis(smooth_data, 1, data)
            
            # Apply existing normalization
            normalized_data, _ = normalize_data(smoothed_data, scaler)
            
            # Apply existing PCA transformation
            pca_data, _ = apply_pca(normalized_data, pca)
            
        else:
            # No existing preprocessing objects, create new ones
            # Smooth data using Savitzky-Golay filter
            smoothed_data = np.apply_along_axis(smooth_data, 1, data)
            
            # Normalize data with new scaler
            normalized_data, scaler = normalize_data(smoothed_data)
            
            # Apply PCA with new PCA object
            pca_data, pca = apply_pca(normalized_data)
            
            # Save preprocessing objects for future use
            os.makedirs(os.path.dirname(preproc_path), exist_ok=True)
            joblib.dump((scaler, pca), preproc_path)
        
        return pca_data
        
    except Exception as e:
        current_app.logger.error(f"Error during preprocessing: {str(e)}")
        # Fallback to basic preprocessing
        
        # Smooth data
        smoothed_data = np.apply_along_axis(smooth_data, 1, data)
        
        # Normalize
        normalized_data, _ = normalize_data(smoothed_data)
        
        return normalized_data
"""
Preprocessing functions for spectral data
"""
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib
import os
from backend.core.config import Config

# Load configuration
config = Config()

def preprocess_data(df):
    """
    Initial preprocessing of raw data
    
    Args:
        df: Pandas DataFrame with raw spectral data
        
    Returns:
        DataFrame: Preprocessed data ready for further processing
    """
    # Make a copy to avoid modifying original data
    processed_df = df.copy()
    
    # Extract feature columns (assuming the first column might be a label or ID)
    if processed_df.columns[0].lower() in ['id', 'label', 'sample', 'class']:
        X = processed_df.iloc[:, 1:].copy()
        labels = processed_df.iloc[:, 0]
    else:
        X = processed_df.copy()
        labels = None
    
    # Convert columns to numeric if possible
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Handle missing values (replace with column mean)
    X = X.fillna(X.mean())
    
    # Return processed features and labels if available
    if labels is not None:
        # Keep labels as first column
        result = pd.concat([labels.reset_index(drop=True), X.reset_index(drop=True)], axis=1)
        return result
    else:
        return X

def apply_savitzky_golay(df):
    """
    Apply Savitzky-Golay filter for smoothing spectral data
    
    Args:
        df: DataFrame with preprocessed spectral data
        
    Returns:
        DataFrame: Filtered data
    """
    # Make a copy
    filtered_df = df.copy()
    
    # Check if first column is non-numeric (label)
    first_col_is_label = not pd.api.types.is_numeric_dtype(filtered_df.iloc[:, 0])
    
    # Apply filter only to numeric columns
    if first_col_is_label:
        label_col = filtered_df.iloc[:, 0]
        numeric_cols = filtered_df.iloc[:, 1:]
        
        # Apply Savitzky-Golay filter to each row of spectral data
        filtered_data = np.apply_along_axis(
            lambda x: savgol_filter(x, config.SAVGOL_WINDOW, config.SAVGOL_POLYORDER),
            axis=1,
            arr=numeric_cols.values
        )
        
        # Rebuild DataFrame
        filtered_numeric = pd.DataFrame(filtered_data, columns=numeric_cols.columns)
        filtered_df = pd.concat([label_col.reset_index(drop=True), 
                                filtered_numeric.reset_index(drop=True)], axis=1)
    else:
        # Apply filter to all columns
        filtered_data = np.apply_along_axis(
            lambda x: savgol_filter(x, config.SAVGOL_WINDOW, config.SAVGOL_POLYORDER),
            axis=1,
            arr=filtered_df.values
        )
        filtered_df = pd.DataFrame(filtered_data, columns=filtered_df.columns)
    
    return filtered_df

def normalize_data(df):
    """
    Normalize spectral data using StandardScaler
    
    Args:
        df: DataFrame with filtered spectral data
        
    Returns:
        DataFrame: Normalized data
    """
    # Make a copy
    normalized_df = df.copy()
    
    # Check if first column is non-numeric (label)
    first_col_is_label = not pd.api.types.is_numeric_dtype(normalized_df.iloc[:, 0])
    
    if first_col_is_label:
        label_col = normalized_df.iloc[:, 0]
        numeric_cols = normalized_df.iloc[:, 1:]
        
        # Apply normalization
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(numeric_cols)
        
        # Save the scaler for later use
        scaler_path = os.path.join(config.MODEL_DIR, 'standard_scaler.pkl')
        joblib.dump(scaler, scaler_path)
        
        # Rebuild DataFrame
        normalized_numeric = pd.DataFrame(normalized_data, columns=numeric_cols.columns)
        normalized_df = pd.concat([label_col.reset_index(drop=True), 
                                 normalized_numeric.reset_index(drop=True)], axis=1)
    else:
        # Apply normalization to all columns
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(normalized_df)
        
        # Save the scaler for later use
        scaler_path = os.path.join(config.MODEL_DIR, 'standard_scaler.pkl')
        joblib.dump(scaler, scaler_path)
        
        normalized_df = pd.DataFrame(normalized_data, columns=normalized_df.columns)
    
    return normalized_df

def apply_pca(df):
    """
    Apply Principal Component Analysis for dimensionality reduction
    
    Args:
        df: DataFrame with normalized spectral data
        
    Returns:
        ndarray: PCA-transformed features ready for model input
    """
    # Check if first column is non-numeric (label)
    first_col_is_label = not pd.api.types.is_numeric_dtype(df.iloc[:, 0])
    
    if first_col_is_label:
        # Extract numeric columns for PCA
        X = df.iloc[:, 1:].values
    else:
        # All columns are numeric
        X = df.values
    
    # Apply PCA
    pca = PCA(n_components=config.PCA_COMPONENTS)
    pca_result = pca.fit_transform(X)
    
    # Save the PCA model for later use
    pca_path = os.path.join(config.MODEL_DIR, 'pca_model.pkl')
    joblib.dump(pca, pca_path)
    
    return pca_result
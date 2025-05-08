# core/utils/validation.py
"""
Validation functions for input data.
"""

import numpy as np
import pandas as pd
from flask import current_app

def allowed_file(filename):
    """Check if the file has an allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']

def validate_csv_content(df):
    """
    Validate the content of a CSV file
    Returns a dict with validation result and preprocessed data if valid
    """
    # Check if dataframe is empty
    if df.empty:
        return {
            'valid': False,
            'message': 'CSV file is empty'
        }
    
    # Check if there are any numeric columns (required for spectral data)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return {
            'valid': False,
            'message': 'No numeric data found in CSV file'
        }
    
    # Identify ID column if exists
    id_column = None
    for col_name in ['id', 'ID', 'sample_id', 'Sample_ID']:
        if col_name in df.columns:
            id_column = col_name
            break
    
    # Select only numeric columns that aren't the ID
    feature_cols = [col for col in numeric_cols if col != id_column]
    
    # Check if there's enough spectral data
    if len(feature_cols) < 3:  # Arbitrary minimum number of features
        return {
            'valid': False,
            'message': 'Not enough numeric columns for spectral analysis'
        }
    
    # Check for missing values
    if df[feature_cols].isna().any().any():
        return {
            'valid': False,
            'message': 'CSV contains missing values in numeric columns'
        }
    
    # Extract the data matrix
    data = df[feature_cols].values
    
    return {
        'valid': True,
        'data': data,
        'id_column': id_column,
        'feature_columns': feature_cols
    }

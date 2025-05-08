"""
Validation functions for input data
"""
import pandas as pd
import numpy as np
import os

def validate_csv(file_path):
    """
    Validate CSV file format and content
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        dict: Validation result with 'valid' boolean and optional 'error' message
    """
    # Check if file exists
    if not os.path.exists(file_path):
        return {"valid": False, "error": "File does not exist"}
    
    # Check file extension
    if not file_path.endswith('.csv'):
        return {"valid": False, "error": "File must be a CSV"}
    
    try:
        # Try to read the CSV file
        df = pd.read_csv(file_path)
        
        # Check if file is empty
        if df.empty:
            return {"valid": False, "error": "CSV file is empty"}
        
        # Check for minimum number of columns (wavelength columns for spectral data)
        if df.shape[1] < 5:  # Arbitrary minimum, adjust based on your requirements
            return {"valid": False, "error": "CSV must contain at least 5 columns of spectral data"}
        
        # Check for numeric data in all columns (except maybe the first one which could be labels)
        non_numeric_cols = []
        for col in df.columns[1:]:  # Skip the first column which might be a label
            if not pd.api.types.is_numeric_dtype(df[col]):
                non_numeric_cols.append(col)
        
        if non_numeric_cols:
            return {"valid": False, "error": f"The following columns contain non-numeric data: {', '.join(non_numeric_cols)}"}
        
        # Check for missing values
        if df.isnull().any().any():
            return {"valid": False, "error": "CSV contains missing values"}
        
        # File passed all validation checks
        return {"valid": True}
        
    except pd.errors.ParserError:
        return {"valid": False, "error": "Failed to parse CSV file"}
    except Exception as e:
        return {"valid": False, "error": f"Validation error: {str(e)}"}
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib
import os
import traceback

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
MODEL_DIR = 'models/trained_model.pkl'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained model
def load_model():
    try:
        model = joblib.load(MODEL_DIR)
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

# Validate the CSV file
def validate_csv(file_path):
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Check if the file is empty
        if df.empty:
            return False, "Empty CSV file"
        
        # Check if the file contains spectral data (assuming specific wavelength columns)
        # This should be adjusted based on your specific data format
        wavelength_columns = [col for col in df.columns if col.replace('.', '').isdigit()]
        if len(wavelength_columns) < 10:  # Assuming at least 10 wavelength columns
            return False, "CSV does not contain enough spectral wavelength columns"
            
        return True, df
    except Exception as e:
        return False, f"Error validating CSV: {str(e)}"

# Preprocess the spectral data
def preprocess_data(df):
    try:
        # Extract wavelength columns (assuming they are numeric)
        wavelength_columns = [col for col in df.columns if col.replace('.', '').isdigit()]
        
        # Convert wavelength columns to float if they aren't already
        spectral_data = df[wavelength_columns].astype(float).values
        
        # 1. Savitzky-Golay filtering for smoothing
        smoothed_data = np.zeros_like(spectral_data)
        for i in range(spectral_data.shape[0]):
            # Parameters (window_length, polyorder) should be adjusted based on your data
            smoothed_data[i] = savgol_filter(spectral_data[i], window_length=11, polyorder=2)
        
        # 2. Normalization (Standard Scaling)
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(smoothed_data)
        
        # 3. PCA for dimensionality reduction
        # Number of components should be adjusted based on your needs
        pca = PCA(n_components=20)  
        reduced_data = pca.fit_transform(normalized_data)
        
        return True, reduced_data, pca, scaler
    except Exception as e:
        return False, f"Error preprocessing data: {str(e)}", None, None

# Make predictions using the trained model
def make_predictions(processed_data, model):
    try:
        # Make predictions for moisture, protein, and fiber content
        predictions = model.predict(processed_data)
        
        # Assuming the model output is structured as [moisture, protein, fiber]
        # If you have multiple models or different structure, adjust accordingly
        return True, predictions
    except Exception as e:
        return False, f"Error making predictions: {str(e)}"

# Format the results as needed
def format_results(predictions, original_df):
    try:
        # Create a dataframe with sample IDs and predictions
        results_df = pd.DataFrame()
        
        # Add sample IDs if available
        if 'sample_id' in original_df.columns:
            results_df['sample_id'] = original_df['sample_id']
        else:
            results_df['sample_id'] = [f"Sample_{i+1}" for i in range(len(predictions))]
        
        # Add predictions
        results_df['moisture'] = predictions[:, 0]
        results_df['protein'] = predictions[:, 1]
        results_df['fiber'] = predictions[:, 2]
        
        # Convert to dictionary for JSON response
        results_dict = {
            'predictions': results_df.to_dict(orient='records'),
            'summary': {
                'moisture': {
                    'mean': float(results_df['moisture'].mean()),
                    'min': float(results_df['moisture'].min()),
                    'max': float(results_df['moisture'].max())
                },
                'protein': {
                    'mean': float(results_df['protein'].mean()),
                    'min': float(results_df['protein'].min()),
                    'max': float(results_df['protein'].max())
                },
                'fiber': {
                    'mean': float(results_df['fiber'].mean()),
                    'min': float(results_df['fiber'].min()),
                    'max': float(results_df['fiber'].max())
                }
            }
        }
        
        return True, results_dict
    except Exception as e:
        return False, f"Error formatting results: {str(e)}"

@app.route('/api/predict', methods=['POST'])
def predict():
    # Check if a file was uploaded
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    # If the user did not select a file
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400
    
    try:
        # Save the uploaded file
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        
        # Step 1: Validate the CSV
        valid, result = validate_csv(file_path)
        if not valid:
            return jsonify({'success': False, 'error': result}), 400
        
        original_df = result
        
        # Step 2: Preprocess the data
        success, processed_data, pca, scaler = preprocess_data(original_df)
        if not success:
            return jsonify({'success': False, 'error': processed_data}), 500
        
        # Step 3: Load the trained model
        model = load_model()
        if model is None:
            return jsonify({'success': False, 'error': 'Failed to load model'}), 500
        
        # Step 4: Make predictions
        success, predictions = make_predictions(processed_data, model)
        if not success:
            return jsonify({'success': False, 'error': predictions}), 500
        
        # Step 5: Format the results
        success, formatted_results = format_results(predictions, original_df)
        if not success:
            return jsonify({'success': False, 'error': formatted_results}), 500
        
        # Step 6: Return the results to the frontend
        return jsonify({'success': True, 'results': formatted_results}), 200
        
    except Exception as e:
        error_message = f"An unexpected error occurred: {str(e)}"
        error_traceback = traceback.format_exc()
        print(error_message)
        print(error_traceback)
        return jsonify({'success': False, 'error': error_message}), 500

if __name__ == '__main__':
    app.run(debug=True)
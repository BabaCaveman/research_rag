# Enhanced Spectral Regression Pipeline

This package provides a comprehensive machine learning pipeline for regression analysis on spectral data. The pipeline includes advanced preprocessing techniques and multiple model options, including CNN (Convolutional Neural Network) specifically optimized for spectral data.

## Features

- **Data Preprocessing**:
  - Savitzky-Golay smoothing for noise reduction
  - Standard scaling (Z-score normalization)
  - Min-Max scaling (for CNN models)
  - PCA dimensionality reduction
  - Configurable preprocessing steps

- **Model Options**:
  - Random Forest Regressor
  - Support Vector Regression (SVR)
  - Convolutional Neural Network (CNN) for spectral data

- **Model Evaluation**:
  - Cross-validation
  - Train/test split evaluation
  - Overfitting detection
  - Detailed performance metrics (R², RMSE, MAE)
  - Prediction visualization

- **Utility Functions**:
  - Model saving and loading
  - Batch prediction on new data
  - Feature importance visualization
  - Learning curve visualization

## Requirements

- Python 3.7+
- numpy
- pandas
- scikit-learn
- scipy
- matplotlib
- joblib
- tensorflow (optional, for CNN models)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training a Model

```bash
python regression_spectral_ml.py --data your_data.csv --target moisture --model CNN --output models
```

#### Command-line Arguments

- `--data`: Path to CSV data file
- `--target`: Name of the target column (e.g., 'moisture', 'protein')
- `--model`: Model type ('RandomForest', 'SVR', or 'CNN')
- `--output`: Directory to save model files
- `--preprocessing`: Preprocessing steps to apply (comma-separated: 'smoothing,scaling,pca')
- `--window`: Window length for Savitzky-Golay filter (must be odd)
- `--polyorder`: Polynomial order for Savitzky-Golay filter
- `--pca_variance`: Variance threshold for PCA
- `--skip_pca_for_cnn`: Flag to skip PCA for CNN models to preserve spatial structure

### Making Predictions

```bash
python spectral_utils.py --data new_data.csv --model_dir models --target moisture --output predictions.csv
```

#### Command-line Arguments

- `--data`: Path to CSV data file for prediction
- `--model_dir`: Directory containing saved model
- `--target`: Name of target column
- `--output`: Path to save predictions CSV

## Example Workflow

### 1. Prepare Data

Ensure your CSV file has spectral data as columns and target variable(s) as a single column.

### 2. Train Model

```bash
# Train Random Forest with default preprocessing
python regression_spectral_ml.py --data spectral_data.csv --target moisture --model RandomForest

# Train CNN with specific preprocessing
python regression_spectral_ml.py --data spectral_data.csv --target moisture --model CNN --preprocessing smoothing,scaling --skip_pca_for_cnn
```

### 3. Make Predictions

```bash
python spectral_utils.py --data new_samples.csv --model_dir models --target moisture --output predictions.csv
```

## CNN Model for Spectral Data

The CNN model is specifically designed for spectral data regression:

- 1D convolutional layers to capture patterns in the spectrum
- Batch normalization for training stability
- Dropout for regularization
- Early stopping to prevent overfitting
- Learning rate scheduling

The model architecture:
```
Input → Conv1D → BatchNorm → MaxPooling → Dropout → Conv1D → BatchNorm → MaxPooling → Dropout → Flatten → Dense → BatchNorm → Dropout → Output
```

## Tips for Best Performance

1. **For CNN Models**:
   - Consider skipping PCA to preserve the spectral structure (`--skip_pca_for_cnn`)
   - Use MinMax scaling instead of standard scaling (automatic when using CNN model)

2. **For Random Forest**:
   - PCA can improve performance by reducing noise
   - Standard scaling is less critical but still recommended

3. **For SVR**:
   - Both PCA and standard scaling are important
   - RBF kernel usually works best for spectral data

4. **Preprocessing**:
   - Savitzky-Golay smoothing parameters should be adjusted based on your specific spectral data
   - Larger window sizes (11-21) are good for NIR data with broad peaks
   - Smaller window sizes (5-9) work better for Raman or FTIR with sharper peaks

## Model Output Files

When a model is trained, the following files are created:

- `spectral_{target}_model.joblib` or `spectral_{target}_cnn_model/`: The trained model
- `preprocessing_{target}.joblib`: Preprocessing objects (scaler, PCA)
- `model_{target}_metadata.joblib`: Metadata about the model
- `{target}_{model_type}_predictions.png`: Visualization of model predictions
- `pca_explained_variance.png`: PCA explained variance plot (if PCA is used)
- `cnn_learning_curve.png`: CNN learning curve plot (if CNN is used)

## License

MIT
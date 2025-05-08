import os
import numpy as np
import pandas as pd
import logging
import joblib
import sys
from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.decomposition import PCA
import pickle

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Check and install required packages if needed
def check_and_install_packages():
    required_packages = {
        'sklearn': 'scikit-learn>=1.0.0',
        'scipy': 'scipy>=1.7.0',
        'pandas': 'pandas>=1.3.0',
        'numpy': 'numpy>=1.20.0',
        'joblib': 'joblib>=1.0.0'
    }
    
    missing_packages = []
    
    for package, install_name in required_packages.items():
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(install_name)
    
    if missing_packages:
        logging.error("Missing required packages. Please install them using:")
        command = f"pip install {' '.join(missing_packages)}"
        logging.error(f"    {command}")
        return False
    
    return True

class SpectralRFPipeline:
    def __init__(self, data_path, target_variable, test_size=0.2, random_state=42):
        self.data_path = data_path
        self.target_variable = target_variable
        self.test_size = test_size
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.pca = None
        self.metadata = None
    
    def load_data(self):
        try:
            logging.info(f"Loading data from {self.data_path}")
            df = pd.read_csv(self.data_path)
            logging.info(f"Dataset shape: {df.shape}")
            
            # Identify column types
            spectral_cols = [col for col in df.columns if col.startswith('R_')]
            metadata_cols = [col for col in df.columns if col in ['Sample_ID', 'Timestamp', 'Crop_Type']]
            target_cols = [col for col in df.columns if col in ['Moisture', 'Fiber', 'Protein']]
            
            if not spectral_cols:
                logging.error("No spectral columns (starting with 'R_') found in the dataset.")
                return None, None
                
            if self.target_variable not in df.columns:
                logging.error(f"Target variable '{self.target_variable}' not found in the dataset.")
                return None, None
            
            # Extract only spectral columns for X
            X = df[spectral_cols]
            y = df[self.target_variable]
            
            # Check if y contains non-numeric values
            if not pd.api.types.is_numeric_dtype(y):
                logging.error(f"Target variable '{self.target_variable}' contains non-numeric values.")
                return None, None
            
            # Keep metadata for reference if needed
            self.metadata_df = df[metadata_cols + target_cols] if metadata_cols else df[target_cols]
            
            # Handle NaN values
            x_nan_count = X.isna().sum().sum()
            y_nan_count = y.isna().sum()
            if x_nan_count > 0:
                logging.warning(f"Found {x_nan_count} NaN values in feature data. Imputing with mean.")
                X.fillna(X.mean(), inplace=True)
            if y_nan_count > 0:
                logging.warning(f"Found {y_nan_count} NaN values in target data. Removing affected samples.")
                non_nan_indices = ~y.isna()
                X = X[non_nan_indices]
                y = y[non_nan_indices]
            
            logging.info(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
            return X, y
            
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            return None, None
    
    def preprocess_data(self, X, smooth=True, window_length=11, polyorder=2, derivative=None, use_pca=False, n_components=None):
        try:
            if X is None:
                return None
                
            # Ensure we're only working with spectral columns
            spectral_cols = [col for col in X.columns if col.startswith('R_')]
            X_spectral = X[spectral_cols].copy()
        
            # Verify data is numeric
            non_numeric_cols = [col for col in spectral_cols if not pd.api.types.is_numeric_dtype(X_spectral[col])]
            if non_numeric_cols:
                logging.error(f"Found non-numeric columns in spectral data: {non_numeric_cols}")
                return None
        
            # Verify window length is appropriate
            if window_length >= X_spectral.shape[1]:
                window_length = min(11, X_spectral.shape[1] - 1)
                if window_length % 2 == 0:  # Must be odd
                     window_length -= 1
                logging.warning(f"Adjusted window length to {window_length}")
        
            # Apply Savitzky-Golay smoothing
            if smooth:
                X_smoothed = np.apply_along_axis(
                    lambda x: savgol_filter(x, window_length=window_length, polyorder=polyorder),
                    axis=1, arr=X_spectral.values
                )
                X_processed = pd.DataFrame(X_smoothed, columns=spectral_cols)
            else:
                X_processed = X_spectral.copy()
        
            # Apply derivatives if requested
            if derivative is not None:
                X_derivative = np.apply_along_axis(
                    lambda x: savgol_filter(x, window_length=window_length, polyorder=polyorder, deriv=derivative),
                    axis=1, arr=X_processed.values
                )
                X_processed = pd.DataFrame(X_derivative, columns=spectral_cols)
        
            # Apply PCA if requested
            if use_pca:
                # Initialize PCA - can be either a specific number of components or a variance threshold
                if isinstance(n_components, float) and 0 < n_components < 1:
                    logging.info(f"Applying PCA to explain {n_components*100:.1f}% of variance")
                    pca = PCA(n_components=n_components)
                else:
                    logging.info(f"Applying PCA to reduce to {n_components} components")
                    pca = PCA(n_components=int(n_components))
            
                # Fit and transform the data
                X_pca = pca.fit_transform(X_processed)
            
                # Log the explained variance
                explained_variance = pca.explained_variance_ratio_.cumsum()
                n_kept = len(explained_variance)
                logging.info(f"PCA reduced dimensions from {X_processed.shape[1]} to {n_kept} components")
                logging.info(f"Total explained variance: {explained_variance[-1]:.4f}")
            
                # Create column names for PCA components
                pca_cols = [f"PC_{i+1}" for i in range(X_pca.shape[1])]
                X_processed = pd.DataFrame(X_pca, columns=pca_cols)
            
                # Store PCA for future use (e.g., transforming new data)
                self.pca = pca
        
            return X_processed
        
        except Exception as e:
            logging.error(f"Error in preprocessing data: {str(e)}")
            return None
    
    def split_data(self, X, y):
        try:
            if X is None or y is None:
                return None, None, None, None
                
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )
            logging.info(f"Data split: {X_train.shape[0]} training samples, {X_test.shape[0]} testing samples")
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logging.error(f"Error splitting data: {str(e)}")
            return None, None, None, None
    
    def scale_data(self, X_train, X_test):
        try:
            if X_train is None or X_test is None:
                return None, None
            
            # Store column names if X_train is a DataFrame
            self.feature_names = X_train.columns if hasattr(X_train, 'columns') else None
                
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            return X_train_scaled, X_test_scaled
            
        except Exception as e:
            logging.error(f"Error scaling data: {str(e)}")
            return None, None
    
    def build_rf_model(self, n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        try:
            logging.info("Building Random Forest model")
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=self.random_state,
                n_jobs=-1  # Use all available cores
            )
            
            self.model = model
            return model
            
        except Exception as e:
            logging.error(f"Error building Random Forest model: {str(e)}")
            return None
    
    def tune_hyperparameters(self, X_train, y_train, cv=5, stratify=False):
        try:
            if X_train is None or y_train is None:
                return None
                
            # Define parameter grid
            param_grid = {
                'n_estimators': [300, 400, 500],
                'max_depth': [6, 8, 10],
                'min_samples_split': [10, 20, 30],
                'min_samples_leaf': [2, 4, 6],
                'max_features': [0.3, 0.5, 0.7]
            }
            
            # Create base model
            rf = RandomForestRegressor(random_state=self.random_state, n_jobs=-1)
            
            # Set up cross-validation strategy
            if stratify:
                # Create bins for stratification (since y is continuous)
                y_binned = pd.qcut(y_train, q=5, labels=False, duplicates='drop')
                cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
                cv_arg = cv_strategy.split(X_train, y_binned)
            else:
                cv_arg = cv
            
            # Create GridSearchCV object
            grid_search = GridSearchCV(
                estimator=rf,
                param_grid=param_grid,
                cv=cv_arg,
                scoring='neg_mean_squared_error',
                verbose=1,
                n_jobs=-1
            )
            
            # Fit GridSearchCV
            grid_search.fit(X_train, y_train)
            
            # Get best parameters and model
            best_params = grid_search.best_params_
            logging.info(f"Best hyperparameters: {best_params}")
            
            # Update model with best parameters
            self.model = grid_search.best_estimator_
            
            return self.model
            
        except Exception as e:
            logging.error(f"Error tuning hyperparameters: {str(e)}")
            return None
    
    def train_model(self, X_train, y_train, k_fold=None, stratify=False):
        try:
            if self.model is None or X_train is None or y_train is None:
                return None
                
            logging.info("Training Random Forest model")
            
            # Train with k-fold cross-validation if requested
            if k_fold is not None:
                logging.info(f"Using {k_fold}-fold cross-validation during training")
                
                # Convert numpy arrays to arrays if needed for consistency
                X_train_arr = X_train
                y_train_arr = y_train
                
                if stratify:
                    # Create bins for stratification (since y is continuous)
                    if hasattr(y_train, 'values'):  # If y_train is a pandas Series
                        y_train_values = y_train.values
                    else:
                        y_train_values = y_train
                        
                    # Create bins for stratification using numpy operations
                    y_binned = pd.qcut(pd.Series(y_train_values), q=5, labels=False, duplicates='drop')
                    cv_strategy = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=self.random_state)
                    folds = list(cv_strategy.split(X_train_arr, y_binned))
                else:
                    cv_strategy = KFold(n_splits=k_fold, shuffle=True, random_state=self.random_state)
                    folds = list(cv_strategy.split(X_train_arr))
                
                # Collect scores from each fold
                fold_scores = []
                for fold, (train_idx, val_idx) in enumerate(folds):
                    # Use array indexing instead of DataFrame iloc
                    X_fold_train = X_train_arr[train_idx]
                    X_fold_val = X_train_arr[val_idx]
                    
                    if hasattr(y_train, 'iloc'):  # If y_train is pandas
                        y_fold_train = y_train.iloc[train_idx]
                        y_fold_val = y_train.iloc[val_idx]
                    else:  # If y_train is numpy array
                        y_fold_train = y_train[train_idx]
                        y_fold_val = y_train[val_idx]
                    
                    self.model.fit(X_fold_train, y_fold_train)
                    y_pred = self.model.predict(X_fold_val)
                    fold_rmse = np.sqrt(mean_squared_error(y_fold_val, y_pred))
                    fold_scores.append(fold_rmse)
                    logging.info(f"Fold {fold+1}/{k_fold} RMSE: {fold_rmse:.4f}")
                
                logging.info(f"CV RMSE: {np.mean(fold_scores):.4f} (±{np.std(fold_scores):.4f})")
            
            # Final training on full dataset
            self.model.fit(X_train, y_train)
            
            # Get feature importances
            importances = self.model.feature_importances_
            feature_names = self.feature_names if self.feature_names is not None else [
                f"feature_{i}" for i in range(X_train.shape[1])
            ]
                
            # Sort feature importances in descending order
            indices = np.argsort(importances)[::-1]
            
            # Log top 10 important features
            logging.info("Top 10 important features:")
            for i in range(min(10, len(indices))):
                logging.info(f"  {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
            
            return self.model
            
        except Exception as e:
            logging.error(f"Error training model: {str(e)}")
            return None
    
    def evaluate_model(self, X_test, y_test, k_fold=None, stratify=False):
        try:
            if self.model is None or X_test is None or y_test is None:
                return None
                
            logging.info("Evaluating Random Forest model performance")
            
            # Direct test set evaluation
            y_pred = self.model.predict(X_test)
            y_test_np = np.array(y_test)
            y_pred_np = np.array(y_pred)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test_np, y_pred_np))
            mae = mean_absolute_error(y_test_np, y_pred_np)
            r2 = r2_score(y_test_np, y_pred_np)
            
            logging.info(f"Test R² score: {r2:.4f}")
            logging.info(f"Test RMSE: {rmse:.4f}")
            logging.info(f"Test MAE: {mae:.4f}")
            
            # K-fold cross-validation if requested
            cv_results = None
            if k_fold is not None:
                logging.info(f"Performing {k_fold}-fold cross-validation")
                
                # For regression tasks, we should use regular KFold, not StratifiedKFold
                # Stratified KFold is only for classification tasks
                cv_strategy = KFold(n_splits=k_fold, shuffle=True, random_state=self.random_state)
                
                cv_scores = cross_val_score(
                    self.model, X_test, y_test, 
                    cv=cv_strategy, 
                    scoring='neg_mean_squared_error',
                    n_jobs=-1
                )
                
                cv_rmse = np.sqrt(-cv_scores)
                logging.info(f"CV RMSE: {cv_rmse.mean():.4f} (±{cv_rmse.std():.4f})")
                
                cv_results = {
                    'cv_rmse_mean': cv_rmse.mean(),
                    'cv_rmse_std': cv_rmse.std(),
                    'cv_scores': cv_rmse
                }
            
            return {
                'r2': r2,
                'rmse': rmse,
                'mae': mae,
                'y_test': y_test_np,
                'y_pred': y_pred_np,
                'cv_results': cv_results
            }
            
        except Exception as e:
            logging.error(f"Error evaluating model: {str(e)}")
            return None
    
    def check_overfitting(self, X_train, y_train):
        try:
            if self.model is None or X_train is None or y_train is None:
                return None
                
            y_train_pred = self.model.predict(X_train)
            
            # Ensure arrays are the right shape
            y_train_np = np.array(y_train)
            y_train_pred_np = np.array(y_train_pred)
            
            train_rmse = np.sqrt(mean_squared_error(y_train_np, y_train_pred_np))
            train_r2 = r2_score(y_train_np, y_train_pred_np)
            
            logging.info(f"Training RMSE: {train_rmse:.4f}")
            logging.info(f"Training R²: {train_r2:.4f}")
            
            return {
                'train_rmse': train_rmse,
                'train_r2': train_r2
            }
            
        except Exception as e:
            logging.error(f"Error checking for overfitting: {str(e)}")
            return None
    
    def save_model(self, model_dir):
        try:
            if self.model is None:
                logging.error("No model available to save")
                return
            
            os.makedirs(model_dir, exist_ok=True)
        
            # Save the model using pickle (PKL format)
            model_path = os.path.join(model_dir, f'spectral_{self.target_variable}_rf_model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
        
            # Save the scaler
            scaler_path = os.path.join(model_dir, f'scaler_{self.target_variable}.pkl')
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
        
            # Save PCA if used
            if self.pca is not None:
                pca_path = os.path.join(model_dir, f'pca_{self.target_variable}.pkl')
                with open(pca_path, 'wb') as f:
                    pickle.dump(self.pca, f)
                logging.info(f"PCA transformer saved to {pca_path}")
        
            # Save metadata about the model and preprocessing
            metadata = {
                'target_variable': self.target_variable,
                'model_path': model_path,
                'scaler_path': scaler_path,
                'model_type': 'RandomForest',
                'n_estimators': self.model.n_estimators,
                'max_depth': self.model.max_depth,
                'min_samples_split': self.model.min_samples_split,
                'pca_used': self.pca is not None
            }
        
            # Add PCA info to metadata if used
            if self.pca is not None:
                metadata['pca_path'] = os.path.join(model_dir, f'pca_{self.target_variable}.pkl')
                metadata['pca_n_components'] = self.pca.n_components_
                metadata['pca_explained_variance'] = float(self.pca.explained_variance_ratio_.sum())
        
            metadata_path = os.path.join(model_dir, f'model_{self.target_variable}_metadata.pkl')
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            logging.info(f"Model artifacts saved to {model_dir}")
        
        except Exception as e:
            logging.error(f"Error saving model: {str(e)}")
            
    def load_model(self, model_dir):
        """Load a previously saved model and its associated artifacts"""
        try:
            metadata_path = os.path.join(model_dir, f'model_{self.target_variable}_metadata.pkl')
            if not os.path.exists(metadata_path):
                logging.error(f"No metadata found at {metadata_path}")
                return False
            
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
        
            # Load model
            model_path = metadata.get('model_path')
            if not os.path.exists(model_path):
                model_path = os.path.join(model_dir, f'spectral_{self.target_variable}_rf_model.pkl')
        
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                logging.info(f"Model loaded from {model_path}")
            else:
                logging.error(f"Model file not found at {model_path}")
                return False
            
            # Load scaler
            scaler_path = metadata.get('scaler_path')
            if not os.path.exists(scaler_path):
                scaler_path = os.path.join(model_dir, f'scaler_{self.target_variable}.pkl')
            
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                logging.info(f"Scaler loaded from {scaler_path}")
            else:
                logging.warning(f"Scaler file not found at {scaler_path}, using default scaler")
                self.scaler = StandardScaler()
            
            # Load PCA if it was used
            if metadata.get('pca_used', False):
                pca_path = metadata.get('pca_path')
                if not os.path.exists(pca_path):
                    pca_path = os.path.join(model_dir, f'pca_{self.target_variable}.pkl')
                
                if os.path.exists(pca_path):
                    with open(pca_path, 'rb') as f:
                        self.pca = pickle.load(f)
                    logging.info(f"PCA transformer loaded from {pca_path}")
                    logging.info(f"PCA components: {self.pca.n_components_}, " 
                            f"explained variance: {self.pca.explained_variance_ratio_.sum():.4f}")
                else:
                    logging.warning(f"PCA file not found at {pca_path}")
        
            # Store metadata
            self.metadata = metadata
            return True
        
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            return False

def main():
    try:
        # Check for required packages
        if not check_and_install_packages():
            sys.exit(1)
        
        # Configuration
        data_path = 'crop_reflectance_dataset.csv'
        target_variable = 'Moisture'
        model_dir = 'models'
        use_stratified_kfold = True  # Set to True to use stratified k-fold for training
        num_folds = 5  # Number of folds for cross-validation
        
        # PCA configuration
        use_pca = True  # Enable PCA
        n_components = 0.95  # Keep components that explain 95% of variance
        
        # Initialize pipeline
        pipeline = SpectralRFPipeline(data_path, target_variable)
        
        # Load data
        X, y = pipeline.load_data()
        if X is None or y is None:
            logging.error("Data loading failed. Exiting.")
            sys.exit(1)
        
        # Preprocess data with PCA
        X_processed = pipeline.preprocess_data(X, smooth=True, window_length=11, polyorder=2, 
                                              use_pca=use_pca, n_components=n_components)
        if X_processed is None:
            logging.error("Data preprocessing failed. Exiting.")
            sys.exit(1)
        
        # Split data
        X_train, X_test, y_train, y_test = pipeline.split_data(X_processed, y)
        if X_train is None:
            logging.error("Data splitting failed. Exiting.")
            sys.exit(1)
        
        # Scale data (not always needed after PCA but can still be useful)
        X_train_scaled, X_test_scaled = pipeline.scale_data(X_train, X_test)
        if X_train_scaled is None:
            logging.error("Data scaling failed. Using unscaled data.")
            X_train_scaled, X_test_scaled = X_train, X_test
        
        # Build Random Forest model with better regularization parameters
        model = pipeline.build_rf_model(
            n_estimators=200, 
            max_depth=8,           # Control tree depth
            min_samples_split=30,  # Require more samples to split
            min_samples_leaf=20    # Require more samples in leaf nodes
        )
        if model is None:
            logging.error("Model building failed. Exiting.")
            sys.exit(1)
        
        # Option 1: Train model with cross-validation
        pipeline.train_model(X_train_scaled, y_train, k_fold=num_folds, stratify=use_stratified_kfold)
        
        # Option 2: Tune hyperparameters with stratified k-fold (uncomment to use)
        # pipeline.tune_hyperparameters(X_train_scaled, y_train, cv=num_folds, stratify=use_stratified_kfold)
        
        # Evaluate model with cross-validation
        metrics = pipeline.evaluate_model(X_test_scaled, y_test, k_fold=num_folds, stratify=False)
        if metrics is None:
            logging.error("Model evaluation failed. Exiting.")
            sys.exit(1)
        
        # Check for overfitting
        train_metrics = pipeline.check_overfitting(X_train_scaled, y_train)
        
        # Save model
        pipeline.save_model(model_dir)
        
        logging.info("Random Forest pipeline completed successfully")
        
    except Exception as e:
        logging.error(f"An unexpected error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
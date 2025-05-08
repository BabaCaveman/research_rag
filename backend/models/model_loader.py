# models/model_loader.py
import pickle
import numpy as np
import pandas as pd
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelLoader:
    def __init__(self, model_path, scaler_path, pca_path, metadata_path):
        """Initialize the ModelLoader with paths to model components."""
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.pca_path = pca_path
        self.metadata_path = metadata_path
        
        # Initialize components as None
        self.model = None
        self.scaler = None
        self.pca = None
        self.metadata = None
        self.expected_feature_count = None
        self.input_wavelengths = None
        self.target_column = None
        
    def load_models(self):
        """Load all model components from files."""
        try:
            # Load random forest model
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                logger.info(f"Model loaded successfully from {self.model_path}")
            else:
                logger.error(f"Model file not found: {self.model_path}")
                return False
                
            # Load scaler
            if os.path.exists(self.scaler_path):
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                logger.info(f"Scaler loaded successfully from {self.scaler_path}")
                # Extract expected feature count from scaler
                if hasattr(self.scaler, 'n_features_in_'):
                    self.expected_feature_count = self.scaler.n_features_in_
                    logger.info(f"Scaler expects {self.expected_feature_count} features")
            else:
                logger.warning(f"Scaler file not found: {self.scaler_path}")
                
            # Load PCA
            if os.path.exists(self.pca_path):
                with open(self.pca_path, 'rb') as f:
                    self.pca = pickle.load(f)
                logger.info(f"PCA loaded successfully from {self.pca_path}")
                # If expected_feature_count is not set, get it from PCA
                if self.expected_feature_count is None and hasattr(self.pca, 'n_features_'):
                    self.expected_feature_count = self.pca.n_features_
                    logger.info(f"PCA expects {self.expected_feature_count} features")
            else:
                logger.warning(f"PCA file not found: {self.pca_path}")
                
            # Load metadata
            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, 'rb') as f:
                    self.metadata = pickle.load(f)
                logger.info(f"Metadata loaded successfully from {self.metadata_path}")
                
                # Extract wavelengths and target column from metadata if available
                if self.metadata and isinstance(self.metadata, dict):
                    self.input_wavelengths = self.metadata.get('input_wavelengths')
                    self.target_column = self.metadata.get('target_column')
                    logger.info(f"Metadata contains info about {len(self.input_wavelengths) if self.input_wavelengths else 0} wavelengths")
                    
                    # If expected_feature_count is not set, infer from wavelengths
                    if self.expected_feature_count is None and self.input_wavelengths:
                        self.expected_feature_count = len(self.input_wavelengths)
            else:
                logger.warning(f"Metadata file not found: {self.metadata_path}")
                
            return True
                
        except Exception as e:
            logger.error(f"Error loading model components: {str(e)}", exc_info=True)
            return False
    
    def preprocess_features(self, df):
        """
        Preprocess the input dataframe to match the expected features.
        
        Parameters:
        df (pd.DataFrame): DataFrame containing spectral data
        
        Returns:
        np.ndarray: Preprocessed features ready for prediction
        """
        logger.info(f"Preprocessing features. Input shape: {df.shape}")
        
        # If we have wavelength information from metadata, use it to select columns
        if self.input_wavelengths:
            # Try to match wavelengths with column names
            available_cols = df.columns.tolist()
            selected_cols = []
            
            for wavelength in self.input_wavelengths:
                # Try different formats that might match the wavelength
                potential_matches = [
                    str(wavelength),
                    f"{wavelength}",
                    f"R_{wavelength}",
                    f"R_{wavelength}nm",
                    f"{wavelength}nm"
                ]
                
                # Find first matching column
                matched = False
                for match in potential_matches:
                    if match in available_cols:
                        selected_cols.append(match)
                        matched = True
                        break
                
                if not matched:
                    logger.warning(f"Could not find column for wavelength {wavelength}")
            
            if selected_cols:
                logger.info(f"Selected {len(selected_cols)} wavelength columns based on metadata")
                # Create a subset with only the required columns
                df_selected = df[selected_cols]
            else:
                logger.warning("No columns matched with expected wavelengths, using original spectral columns")
                df_selected = df
        else:
            # Without wavelength info, use all columns
            df_selected = df
            
        # Handle missing values if any
        df_selected = df_selected.fillna(df_selected.mean())
        
        # Convert to numpy array
        X = df_selected.values
        
        # Apply PCA first if available
        if self.pca is not None:
            try:
                # If input features don't match what PCA expects, handle appropriately
                if X.shape[1] != self.pca.n_features_:
                    logger.warning(f"Feature count mismatch! PCA expects {self.pca.n_features_} but got {X.shape[1]}")
                    
                    # OPTION 1: If we have more features than needed, select the first n features
                    if X.shape[1] > self.pca.n_features_:
                        logger.info(f"Selecting first {self.pca.n_features_} features")
                        X = X[:, :self.pca.n_features_]
                    
                    # OPTION 2: If we have fewer features than needed, pad with zeros
                    elif X.shape[1] < self.pca.n_features_:
                        logger.info(f"Padding features from {X.shape[1]} to {self.pca.n_features_}")
                        padding = np.zeros((X.shape[0], self.pca.n_features_ - X.shape[1]))
                        X = np.hstack((X, padding))
                
                logger.info(f"Applying PCA. Input shape: {X.shape}")
                X = self.pca.transform(X)
                logger.info(f"PCA applied. Output shape: {X.shape}")
            except Exception as e:
                logger.error(f"Error applying PCA: {str(e)}", exc_info=True)
                # If PCA fails, continue with original features
                pass
        
        # Apply scaling if scaler is available
        if self.scaler is not None:
            try:
                # Handle dimension mismatch with scaler
                if hasattr(self.scaler, 'n_features_in_') and X.shape[1] != self.scaler.n_features_in_:
                    logger.warning(f"Feature count mismatch! Scaler expects {self.scaler.n_features_in_} but got {X.shape[1]}")
                    
                    # OPTION 1: If we have more features than needed, select the first n features
                    if X.shape[1] > self.scaler.n_features_in_:
                        logger.info(f"Selecting first {self.scaler.n_features_in_} features")
                        X = X[:, :self.scaler.n_features_in_]
                    
                    # OPTION 2: If we have fewer features than needed, pad with zeros
                    elif X.shape[1] < self.scaler.n_features_in_:
                        logger.info(f"Padding features from {X.shape[1]} to {self.scaler.n_features_in_}")
                        padding = np.zeros((X.shape[0], self.scaler.n_features_in_ - X.shape[1]))
                        X = np.hstack((X, padding))
                
                logger.info(f"Applying scaling. Input shape: {X.shape}")
                X = self.scaler.transform(X)
                logger.info(f"Scaling applied. Output shape: {X.shape}")
            except Exception as e:
                logger.error(f"Error applying scaler: {str(e)}", exc_info=True)
                # If scaling fails, continue with unscaled features
                pass
                
        return X
    
    def predict(self, df):
        """
        Make predictions using the loaded model.
        
        Parameters:
        df (pd.DataFrame or np.ndarray): Input data for prediction
        
        Returns:
        np.ndarray: Predicted values
        """
        try:
            # If input is DataFrame, preprocess it
            if isinstance(df, pd.DataFrame):
                logger.info(f"Preprocessing DataFrame with shape {df.shape}")
                X = self.preprocess_features(df)
            else:
                # If already preprocessed numpy array, use directly
                X = df
                
            # Ensure model is loaded
            if self.model is None:
                logger.error("Model not loaded. Call load_models() first.")
                raise ValueError("Model not loaded")
                
            # Make predictions
            logger.info(f"Making predictions with input shape: {X.shape}")
            predictions = self.model.predict(X)
            logger.info(f"Predictions made successfully: {predictions.shape if isinstance(predictions, np.ndarray) else 'scalar'}")
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}", exc_info=True)
            raise
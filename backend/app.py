#!/usr/bin/env python
import os
import sys
import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import io

# Import your configuration
try:
    from core.config import Config
except ImportError:
    try:
        from backend.core.config import Config
    except ImportError:
        print("ERROR: Cannot import Config, but proceeding anyway.")

# Import model loader
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.model_loader import ModelLoader

# Import API router
from api.endpoints import router as api_router

# Initialize FastAPI app
app = FastAPI(title="Moisture Content Prediction API")

# Determine frontend URL based on environment
frontend_url = os.getenv("FRONTEND_URL", "http://localhost:8000")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_url, "http://localhost:3000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model loader with correct paths based on your directory structure
model_loader = ModelLoader(
    model_path="./models/spectral_Moisture_rf_model.pkl",
    scaler_path="./models/scaler_Moisture.pkl",
    pca_path="./models/pca_Moisture.pkl",
    metadata_path="./models/model_Moisture_metadata.pkl"
)

# Include your API router
app.include_router(api_router)

@app.get("/")
async def root():
    """Root endpoint to check if the API is running."""
    return {"message": "Moisture Content Prediction API is running"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model_loader.is_model_loaded(),
    }

@app.get("/api/health")
async def api_health_check():
    """API health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model_loader.is_model_loaded(),
    }

@app.post("/predict")
async def predict_moisture(file: UploadFile = File(...)):
    """
    Predict moisture content from CSV data.
    
    Upload a CSV file with spectral data for prediction.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted")
    
    try:
        # Read the uploaded file
        contents = await file.read()
        data = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Process the data and make predictions
        predictions = model_loader.predict(data)
        
        return {
            "filename": file.filename,
            "predictions": predictions.tolist() if isinstance(predictions, np.ndarray) else predictions,
            "mean_moisture": float(np.mean(predictions)) if len(predictions) > 0 else None,
            "records_processed": len(predictions)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.post("/predict-moisture")
async def predict_moisture_alias(file: UploadFile = File(...)):
    """Alias for the predict endpoint."""
    return await predict_moisture(file)

@app.post("/api/predict-moisture")
async def api_predict_moisture(file: UploadFile = File(...)):
    """API version of the predict endpoint."""
    return await predict_moisture(file)

def main():
    # Load the models
    print("Loading prediction models...")
    model_loader.load_models()
    print("Models loaded successfully!")
    
    # Run the server
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    print(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    main()
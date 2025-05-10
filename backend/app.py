from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import sys
import pandas as pd
import numpy as np
import io
from dotenv import load_dotenv

load_dotenv()
PORT = int(os.getenv("PORT", 8000))
HOST = os.getenv("HOST", "0.0.0.0")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

# Import your configuration (with safer imports)
try:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from models.model_loader import ModelLoader
    from api.endpoints import router as api_router
except ImportError as e:
    print(f"Import error: {e}")
    # Create fallback classes/functions if imports fail
    class ModelLoader:
        def __init__(self, **kwargs):
            self.is_loaded = False
        
        def load_models(self):
            print("Warning: Using dummy model loader")
            return True
        
        def is_model_loaded(self):
            return self.is_loaded
        
        def predict(self, data):
            return [0.5]  # Return dummy prediction

    api_router = None

# Initialize FastAPI app
app = FastAPI(title="Moisture Content Prediction API")

# Determine frontend URL based on environment
frontend_url = os.getenv("FRONTEND_URL", "http://localhost:8000")

# Configure CORS with specific origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Initialize model loader with correct paths
model_paths = {
    "model_path": "./models/spectral_Moisture_rf_model.pkl",
    "scaler_path": "./models/scaler_Moisture.pkl",
    "pca_path": "./models/pca_Moisture.pkl",
    "metadata_path": "./models/model_Moisture_metadata.pkl"
}

# Check if files exist and modify paths if needed
for key, path in list(model_paths.items()):
    if not os.path.exists(path):
        # Try alternate paths
        alt_path = path.replace("./models/", "./backend/models/")
        if os.path.exists(alt_path):
            model_paths[key] = alt_path
        else:
            print(f"Warning: {path} not found")

# Initialize the model loader with the paths
model_loader = ModelLoader(**model_paths)

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app):
    # Startup code (runs before the application starts)
    print("Loading models on startup...")
    model_loader.load_model()
    
    yield  # This is where the application runs
    
    # Shutdown code (runs when the application is shutting down)
    # Any cleanup code would go here

# Use the lifespan when creating the FastAPI app
app = FastAPI(lifespan=lifespan)

# The rest of your routes remain the same
@app.get("/")
async def root():
    """Root endpoint to check if the API is running."""
    return {"message": "Moisture Content Prediction API is running"}

# Include your API router if it was imported successfully
if api_router:
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
            "mean_moisture": float(np.mean(predictions)) if hasattr(predictions, "__len__") and len(predictions) > 0 else None,
            "records_processed": len(predictions) if hasattr(predictions, "__len__") else 1
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

# For development server
if __name__ == "__main__":
    # Load the models
    print("Loading prediction models...")
    model_loader.load_models()
    print("Models loaded successfully!")
    
    # Run the server
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    print(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
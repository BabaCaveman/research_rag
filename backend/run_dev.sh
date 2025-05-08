#!/bin/bash

# Set the Python path to include the project root directory
# This assumes run_dev.sh is in the backend directory
export PYTHONPATH="$(pwd)/..:$PYTHONPATH"

echo "Setting PYTHONPATH to include parent directory: $PYTHONPATH"

# Check if model files exist
echo "Checking for model files in ./models..."
if [ -d "./models" ]; then
    echo "Model directory exists at: ./models"
    echo "Files in model directory:"
    ls -la ./models
    
    # Check for specific model files
    MODEL_FILES=(
        "./models/spectral_Moisture_rf_model.pkl"
        "./models/scaler_Moisture.pkl"
        "./models/pca_Moisture.pkl"
        "./models/model_Moisture_metadata.pkl"
    )
    
    MISSING_FILES=false
    for FILE in "${MODEL_FILES[@]}"; do
        if [ ! -f "$FILE" ]; then
            echo "Warning: $(basename "$FILE") not found at $FILE"
            MISSING_FILES=true
        else
            echo "Found: $(basename "$FILE")"
        fi
    done
    
    # If required files are missing in models, check if they exist in models/saved
    if [ "$MISSING_FILES" = true ] && [ -d "./models/saved" ]; then
        echo "Checking models/saved directory for model files..."
        
        # Copy files from saved directory if they exist there
        for FILE in "${MODEL_FILES[@]}"; do
            BASENAME=$(basename "$FILE")
            if [ -f "./models/saved/$BASENAME" ]; then
                echo "Found $BASENAME in saved directory, copying to models directory..."
                cp "./models/saved/$BASENAME" "./models/"
                echo "Copied $BASENAME to models directory"
            fi
        done
    fi
else
    echo "Model directory not found at ./models"
    exit 1
fi

# Start the development server
echo "Starting development server..."
python app.py
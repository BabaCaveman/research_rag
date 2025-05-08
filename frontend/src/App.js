import React, { useState, useEffect } from 'react';
import logo from './radar.svg';
import './App.css';
import DataProcessor from './components/dataprocessor';
import ImageUploader from './components/ImageUploader';
import PredictionResult from './components/PredictionResult';
import ApiStatus from './components/ApiStatus';
import { checkApiStatus } from './services/api';

function App() {
  console.log("App component is rendering!");
  
  // State for API connection status
  const [apiConnected, setApiConnected] = useState(false);
  // State for prediction results
  const [predictionResult, setPredictionResult] = useState(null);
  // State to track if prediction is in progress
  const [isLoading, setIsLoading] = useState(false);

  // Function to discover API endpoints
  const discoverApiEndpoints = async () => {
    try {
      // You can implement actual endpoint discovery logic here
      // For example, making a request to a discovery endpoint
      console.log("Attempting to discover API endpoints...");
      
      // Example implementation (replace with actual API call):
      // const response = await fetch('/api/discover');
      // const endpoints = await response.json();
      // console.log("Available endpoints:", endpoints);
    } catch (error) {
      console.error("Failed to discover API endpoints:", error);
    }
  };

  // Check API connection on component mount
  useEffect(() => {
    const verifyApiConnection = async () => {
      const isConnected = await checkApiStatus();
      setApiConnected(isConnected);

      // Add this to debug available endpoints
      console.log("Discovering available API endpoints...");
      await discoverApiEndpoints();
    };
    
    verifyApiConnection();
    // Re-check every 30 seconds
    const intervalId = setInterval(verifyApiConnection, 30000);
    
    return () => clearInterval(intervalId);
  }, []);

  // Handle image selection completion
  const handlePredictionComplete = (result) => {
    setPredictionResult(result);
    setIsLoading(false);
  };
  
  // Handle when prediction starts
  const handlePredictionStart = () => {
    setIsLoading(true);
  };

  return (
    <div className="App">
      <DataProcessor 
        onPredictionStart={handlePredictionStart}
        onPredictionComplete={handlePredictionComplete}
        apiConnected={apiConnected}
      />
      <header className="App-header">
        <div className="api-status-container">
          <ApiStatus connected={apiConnected} />
        </div>
        <img src={logo} className="App-logo" alt="logo" />
        
        {/* Instructions Box - Left Aligned */}
        <div className="text-box text-box-left">
          <h3>Instructions</h3>
          <p>Upload unlabelled spectral data, and the software will predict the relevant crop parameters for you.</p>
          {!apiConnected && (
            <div className="api-warning">
              Backend API is currently unavailable. Please ensure the server is running.
            </div>
          )}
          
          {/* Image uploader component for direct image upload */}
          <div className="image-upload-container">
            <h4>Direct Image Upload</h4>
            <ImageUploader 
              onImageSelected={(file) => {
                // This will be handled by DataProcessor component
                console.log("Image selected:", file?.name);
              }}
              isLoading={isLoading}
            />
          </div>
        </div>
        
        {/* Results Box - Right Aligned */}
        <div className="text-box text-box-right">
          <h3>Results Summary</h3>
          {isLoading ? (
            <div className="loading-indicator">
              <p>Processing your data...</p>
              <div className="spinner"></div>
            </div>
          ) : predictionResult ? (
            <PredictionResult prediction={predictionResult} />
          ) : (
            <p>Your processed data will be summarized here after uploading and processing.</p>
          )}
        </div>
      </header>
    </div>
  );
}

export default App;
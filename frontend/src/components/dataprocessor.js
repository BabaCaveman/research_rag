import React, { useState } from 'react';
import './dataprocessor.css';

function DataProcessor({ onPredictionStart, onPredictionComplete, apiConnected = true }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (!file) return;

    // Reset any previous errors
    setError(null);
    
    // Validate file type - accept images and CSV files for spectral data
    const validTypes = ['image/jpeg', 'image/png', 'image/jpg', 'text/csv', 'application/vnd.ms-excel'];
    if (!validTypes.includes(file.type)) {
      setError('Please select a valid image file (JPEG, PNG) or CSV file.');
      return;
    }

    setSelectedFile(file);
    console.log(`File selected: ${file.name}`);
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setError('Please select a file first.');
      return;
    }

    // Check if API is connected before proceeding
    if (!apiConnected) {
      setError('Cannot connect to the prediction API. Please ensure the backend server is running.');
      return;
    }

    setIsLoading(true);
    setError(null);
    setResults(null);

    // Notify parent component that prediction has started
    if (onPredictionStart) onPredictionStart();

    // Create a FormData object to send the file
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      let endpoint = '/api/process-csv';
      
      // Choose endpoint based on file type
      if (selectedFile.type.startsWith('image/')) {
        endpoint = '/api/predict-moisture'; // Endpoint for image processing
      }

      const response = await fetch(endpoint, {
        method: 'POST',
        body: formData,
      });

      // Check for non-JSON responses first
      const contentType = response.headers.get('content-type');
      if (!contentType || !contentType.includes('application/json')) {
        // Handle HTML or other non-JSON responses
        const textResponse = await response.text();
        throw new Error(`Server returned non-JSON response: ${response.status} - ${response.statusText}`);
      }

      // Now parse as JSON once we've confirmed it's JSON
      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.message || `Server error: ${response.status}`);
      }

      setResults(data);
      
      // Pass the result to parent component
      if (onPredictionComplete) onPredictionComplete(data);
    } catch (err) {
      console.error('Error during file processing:', err);
      
      // More descriptive error message
      if (err.message.includes('SyntaxError') || err.message.includes('Unexpected token')) {
        setError('The server response was not in the expected format. Please check that the API is working correctly.');
      } else {
        setError(err.message || 'Failed to process file. Please try again.');
      }
      
      // Notify parent that prediction failed
      if (onPredictionComplete) onPredictionComplete(null);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="data-processor">
      <h2>Paratus Moisture Prediction</h2>

      {/* File Upload Component */}
      <div className="file-input-container">
        <div className="custom-file-input">
          <input 
            type="file" 
            id="file-input" 
            accept="image/*,text/csv,application/vnd.ms-excel" 
            onChange={handleFileChange} 
            className="hidden-file-input"
            disabled={isLoading}
          />
          <label htmlFor="file-input" className="file-input-label">
            <span className="file-button">Choose File</span>
            <span className="file-name">
              {selectedFile ? selectedFile.name : "No file selected"}
            </span>
          </label>
        </div>
      </div>

      {/* Upload Button */}
      <div className="button-container">
        <button 
          className="upload-button"
          onClick={handleUpload} 
          disabled={isLoading || !selectedFile || !apiConnected}
        >
          {isLoading ? 'Processing...' : 'Analyze File'}
        </button>
      </div>

      {/* Loading Indicator */}
      {isLoading && <p className="loading-indicator">Processing data, please wait...</p>}

      {/* Results Display */}
      {results && (
        <div className="results-container">
          <h3>Prediction Results:</h3>
          {/* Display moisture content prominently if available */}
          {results.moisture_content && (
            <div className="moisture-result">
              <h4>Predicted Moisture Content:</h4>
              <p className="moisture-value">{results.moisture_content.toFixed(2)}%</p>
            </div>
          )}
          
          {/* Display details or full results */}
          {/*<pre>{JSON.stringify(results, null, 2)}</pre>*/}
          
          {/* Table for structured data if applicable */}
          {Array.isArray(results.data) && results.data.length > 0 && (
            <table className="results-table">
              <thead>
                <tr>
                  {Object.keys(results.data[0]).map(key => (
                    <th key={key}>{key}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {results.data.map((row, index) => (
                  <tr key={index}>
                    {Object.values(row).map((value, i) => (
                      <td key={i}>{value}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}

      {/* Error Handling Display */}
      {error && <p className="error-message">Error: {error}</p>}
      
      {/* Instructions */}
      <div className="instructions">
        <p>
          <strong>Supported file types:</strong> Images (JPEG, PNG) or CSV files containing spectral data.
        </p>
        <p>
          <strong>For best results:</strong> Ensure images are clear and well-lit, and CSV files follow the required format.
        </p>
      </div>
    </div>
  );
}

export default DataProcessor;
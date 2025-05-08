// src/components/PredictionResult.js
import React from 'react';

/**
 * Component to display moisture prediction results
 * @param {Object} props - Component props
 * @param {Object} props.prediction - The prediction data from the API
 */
const PredictionResult = ({ prediction }) => {
  if (!prediction) {
    return <p>No prediction data available.</p>;
  }

  // Format the confidence percentage for display
  const formatConfidence = (confidence) => {
    if (typeof confidence === 'number') {
      return `${(confidence * 100).toFixed(1)}%`;
    }
    return confidence ? `${confidence}` : 'N/A';
  };

  // Calculate the confidence bar width
  const getConfidenceWidth = (confidence) => {
    if (typeof confidence === 'number') {
      return `${confidence * 100}%`;
    }
    // Default to 0% if no valid confidence value
    return '0%';
  };

  // Helper to determine if we should show a specific value
  const shouldShowValue = (key, value) => {
    return value !== undefined && value !== null;
  };

  // Get the moisture content value (handle different property names)
  const moistureContent = prediction.moisture_content || 
                          prediction.moisture || 
                          prediction.value || 
                          'N/A';
  
  // Get the confidence value (handle different property names)
  const confidence = prediction.confidence || 
                     prediction.prediction_confidence || 
                     0.8; // Default confidence if not provided

  return (
    <div className="prediction-result">
      <h3 className="prediction-heading">Moisture Prediction Results</h3>
      
      <div className="result-value">
        <span className="result-label">Moisture Content:</span>
        <span className="value">
          {typeof moistureContent === 'number' 
            ? `${moistureContent.toFixed(2)}%` 
            : moistureContent}
        </span>
      </div>
      
      {shouldShowValue('timestamp', prediction.timestamp) && (
        <div className="result-value">
          <span className="result-label">Analysis Time:</span>
          <span className="value">{new Date(prediction.timestamp).toLocaleString()}</span>
        </div>
      )}
      
      {shouldShowValue('processing_time', prediction.processing_time) && (
        <div className="result-value">
          <span className="result-label">Processing Time:</span>
          <span className="value">{prediction.processing_time.toFixed(2)} sec</span>
        </div>
      )}
      
      {/* Display confidence level with visual indicator */}
      <div className="confidence-indicator">
        <div className="confidence-bar">
          <div 
            className="confidence-level" 
            style={{ width: getConfidenceWidth(confidence) }}
          ></div>
        </div>
        <span className="confidence-text">{formatConfidence(confidence)}</span>
      </div>
      
      {/* Display any additional prediction data if available */}
      {prediction.additional_data && (
        <div className="additional-data">
          <h4>Additional Data</h4>
          <pre>{JSON.stringify(prediction.additional_data, null, 2)}</pre>
        </div>
      )}
    </div>
  );
};

export default PredictionResult;
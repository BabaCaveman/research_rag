// src/hooks/usePrediction.js

import { useState } from 'react';
import { predictMoisture } from '../services/api';

export const usePrediction = () => {
  const [prediction, setPrediction] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const predict = async (imageFile) => {
    setIsLoading(true);
    setError(null);
    
    try {
      const result = await predictMoisture(imageFile);
      setPrediction(result);
      return result;
    } catch (err) {
      setError(err.message || 'Failed to get prediction');
      return null;
    } finally {
      setIsLoading(false);
    }
  };

  return {
    prediction,
    isLoading,
    error,
    predict,
  };
};
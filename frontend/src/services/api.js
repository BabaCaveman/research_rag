// services/api.js

/**
 * Check if the API is up and running
 * @returns {Promise<boolean>} True if the API is connected
 */
export const checkApiStatus = async () => {
  try {
    // Try multiple potential API endpoints to increase chances of success
    const potentialPaths = [
      '/health',
      '/api/health',
      '/',
    ];
    
    // Determine the base URL - for development it's likely a different port
    const baseUrl = process.env.REACT_APP_API_URL || 'https://research-rag.onrender.com'; // Updated to match your backend port
    
    // Try each potential path until one works
    for (const path of potentialPaths) {
      try {
        console.log(`Trying API health check at: ${baseUrl}${path}`);
        const response = await fetch(`${baseUrl}${path}`, { 
          method: 'GET',
          headers: {
            'Accept': 'application/json',
          },
          // If you're running in development mode, include credentials
          credentials: 'include',
          // Set a short timeout so we don't hang for too long
          signal: AbortSignal.timeout(3000)
        });
        
        console.log(`API check at ${path} returned status:`, response.status);
        
        // If we get a 200 response, we'll consider it connected
        if (response.ok) {
          return true;
        }
      } catch (pathError) {
        // Just log the error for this path and try the next one
        console.log(`Path ${path} check failed:`, pathError.message);
      }
    }
    
    // If we've tried all paths and none worked, return false
    console.error('All API health check paths failed');
    return false;
  } catch (error) {
    console.error('API connection error:', error);
    return false;
  }
};

/**
 * Upload a CSV file for moisture prediction
 * @param {File} file - The CSV file to upload
 * @returns {Promise<Object>} The prediction results
 */
export const predictMoisture = async (file) => {
  try {
    const baseUrl = process.env.REACT_APP_API_URL || 'https://research-rag.onrender.com';
    const endpoints = ['/predict', '/predict-moisture', '/api/predict-moisture'];
    
    // Create a FormData instance to send the file
    const formData = new FormData();
    formData.append('file', file);
    
    // Try each potential endpoint
    for (const endpoint of endpoints) {
      try {
        console.log(`Attempting to upload to: ${baseUrl}${endpoint}`);
        
        const response = await fetch(`${baseUrl}${endpoint}`, {
          method: 'POST',
          credentials: 'include',
          body: formData,
          // Don't set Content-Type header - the browser will set it correctly with the boundary
          signal: AbortSignal.timeout(30000) // Longer timeout for file uploads
        });
        
        if (response.ok) {
          const data = await response.json();
          console.log('Prediction successful:', data);
          return data;
        }
        
        console.log(`Upload to ${endpoint} failed with status: ${response.status}`);
      } catch (endpointError) {
        console.log(`Upload to ${endpoint} error:`, endpointError.message);
      }
    }
    
    throw new Error('All prediction endpoints failed');
  } catch (error) {
    console.error('Prediction error:', error);
    throw error;
  }
};

/**
 * Discover available API endpoints by checking common paths
 * This can help diagnose API endpoint configuration issues
 */
export const discoverApiEndpoints = async () => {
  const baseUrl = process.env.REACT_APP_API_URL || 'https://research-rag.onrender.com'; // Updated port
  const potentialEndpoints = [
    '/',
    '/api',
    '/api/health',
    '/health'
    // Removed POST endpoints from GET discovery
  ];
  
  const results = {};
  
  // Try all potential endpoints
  for (const endpoint of potentialEndpoints) {
    try {
      const fullUrl = `${baseUrl}${endpoint}`;
      console.log(`Trying endpoint: ${fullUrl}`);
      
      const response = await fetch(fullUrl, { 
        method: 'GET',
        credentials: 'include',
        signal: AbortSignal.timeout(3000)
      });
      
      results[endpoint] = {
        status: response.status,
        statusText: response.statusText,
        contentType: response.headers.get('content-type')
      };
    } catch (error) {
      results[endpoint] = { error: error.message };
    }
  }
  
  console.table(results);
  return results;
};
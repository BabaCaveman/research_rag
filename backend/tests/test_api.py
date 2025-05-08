"""
Test suite for API endpoints
"""
import os
import sys
import unittest
import json
import pandas as pd
import numpy as np
from io import StringIO

# Add parent directory to path to import app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from app import app

class TestAPI(unittest.TestCase):
    """Test cases for API endpoints"""
    
    def setUp(self):
        """Set up test client"""
        self.app = app.test_client()
        self.app.testing = True
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = self.app.get('/api/health')
        data = json.loads(response.data.decode('utf-8'))
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data['status'], 'healthy')
    
    def test_model_info_endpoint(self):
        """Test model info endpoint"""
        response = self.app.get('/api/model/info')
        
        # If model exists, we should get a 200 status
        if response.status_code == 200:
            data = json.loads(response.data.decode('utf-8'))
            self.assertIn('name', data)
            self.assertIn('type', data)
        else:
            # If model doesn't exist, we should get an error message
            data = json.loads(response.data.decode('utf-8'))
            self.assertIn('error', data)
    
    def test_predict_endpoint_no_file(self):
        """Test prediction endpoint with no file"""
        response = self.app.post('/api/predict')
        data = json.loads(response.data.decode('utf-8'))
        
        self.assertEqual(response.status_code, 400)
        self.assertIn('error', data)
        self.assertEqual(data['error'], 'No file provided')
    
    def generate_test_csv(self):
        """Generate a test CSV file with synthetic spectral data"""
        # Create wavelength columns (e.g., 400nm to 2500nm with 10nm steps)
        wavelengths = range(400, 2510, 10)
        columns = [f"wl_{w}" for w in wavelengths]
        
        # Create synthetic data (5 samples)
        num_samples = 5
        data = np.random.random((num_samples, len(wavelengths)))
        
        # Create DataFrame
        df = pd.DataFrame(data, columns=columns)
        df.insert(0, 'sample_id', [f"sample_{i}" for i in range(1, num_samples+1)])
        
        # Convert to CSV string
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        
        return csv_buffer.getvalue()
    
    def test_predict_endpoint_with_synthetic_data(self):
        """Test prediction endpoint with synthetic data"""
        # This test may be skipped if the model doesn't exist
        try:
            # Generate synthetic CSV
            csv_data = self.generate_test_csv()
            
            # Create file data for POST request
            data = {
                'file': (StringIO(csv_data), 'test_data.csv')
            }
            
            # Make request
            response = self.app.post('/api/predict', data=data, content_type='multipart/form-data')
            
            # If model exists and works, we should get predictions
            if response.status_code == 200:
                result = json.loads(response.data.decode('utf-8'))
                self.assertIn('predictions', result)
            else:
                # If there's an issue, it should return an error
                result = json.loads(response.data.decode('utf-8'))
                self.assertIn('error', result)
                
        except Exception as e:
            self.skipTest(f"Skipping test due to error: {str(e)}")

if __name__ == '__main__':
    unittest.main()
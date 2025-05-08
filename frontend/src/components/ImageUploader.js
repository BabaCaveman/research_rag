import React, { useState } from 'react';
import './ImageUploader.css';

const ImageUploader = ({ onImageUpload }) => {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setError(null);
    
    if (!selectedFile) {
      setFile(null);
      setPreview(null);
      return;
    }

    // Check if file is an image
    if (!selectedFile.type.match('image.*')) {
      setError('Please select an image file (PNG, JPG, JPEG, etc)');
      setFile(null);
      setPreview(null);
      return;
    }

    // Create a preview
    const reader = new FileReader();
    reader.onload = () => {
      setPreview(reader.result);
    };
    reader.readAsDataURL(selectedFile);
    
    setFile(selectedFile);
  };

  const handleUpload = async () => {
    if (!file) {
      setError('Please select a file first');
      return;
    }

    setIsUploading(true);
    setError(null);

    try {
      // Create a FormData object to send the file
      const formData = new FormData();
      formData.append('file', file);

      // Get the API URL from environment variables or use default
      const apiUrl = process.env.REACT_APP_API_URL || 'http://localhost:5000';
      
      // Send the file to the backend
      const response = await fetch(`${apiUrl}/api/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Upload failed with status: ${response.status}`);
      }

      const data = await response.json();
      
      // Call the parent component's callback with the upload result
      if (onImageUpload) {
        onImageUpload(data);
      }
      
      // Reset the form after successful upload
      setFile(null);
      setPreview(null);
    } catch (err) {
      setError(`Upload failed: ${err.message}`);
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="image-uploader">
      <div className="upload-container">
        <input
          type="file"
          id="file-input"
          accept="image/*"
          onChange={handleFileChange}
          className="file-input"
        />
        <label htmlFor="file-input" className="file-label">
          {preview ? 'Change Image' : 'Select Image'}
        </label>
        
        <button 
          onClick={handleUpload} 
          disabled={!file || isUploading} 
          className="upload-button"
        >
          {isUploading ? 'Uploading...' : 'Upload'}
        </button>
      </div>

      {error && <div className="error-message">{error}</div>}
      
      {preview && (
        <div className="image-preview">
          <img src={preview} alt="Preview" />
        </div>
      )}
    </div>
  );
};

export default ImageUploader;
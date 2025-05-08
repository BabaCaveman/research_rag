// src/components/ApiStatus.js
import React from 'react';

/**
 * Component to display API connection status
 * @param {Object} props - Component props
 * @param {boolean} props.connected - Whether the API is connected
 */
const ApiStatus = ({ connected }) => {
  return (
    <div className={`api-status ${connected ? 'connected' : 'disconnected'}`}>
      {connected ? 'API Connected' : 'API Disconnected'}
    </div>
  );
};

export default ApiStatus;
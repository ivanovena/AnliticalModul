// src/index.js
import React from 'react';
import ReactDOM from 'react-dom';
import App from './App';
import './index.css';

// Inicializar servicios mock solo si está configurado específicamente
if (process.env.REACT_APP_USE_MOCK_DATA === 'true') {
  const { initMockServices } = require('./mocks/mockService');
  initMockServices();
  console.log('Running with mock data');
} else {
  console.log('Running with real API data');
}

ReactDOM.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
  document.getElementById('root')
);
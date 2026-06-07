/**
 * Application Entry Point
 * =======================
 * 
 * This is the main entry point for the React application. It:
 * - Renders the root App component
 * - Mounts it to the DOM element with id="root"
 * - Applies global CSS styles
 * 
 * Note: React.StrictMode is intentionally removed to prevent double-mounting
 * in development, which would cause the camera to stop unexpectedly.
 */

import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'
import './index.css'

// Create root and render App component
// StrictMode is disabled to prevent camera stream interruption in development
ReactDOM.createRoot(document.getElementById('root')).render(
  <App />
)


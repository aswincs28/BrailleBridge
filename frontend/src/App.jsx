/**
 * Braille Reader - Main Application Component
 * ============================================
 * 
 * This is the root component of the Braille Reader application. It manages:
 * - Tab switching between Camera and Upload modes
 * - Image upload and processing
 * - Recognition results display
 * - Text-to-speech output
 * - Error handling and loading states
 * 
 * Architecture:
 * - Uses React hooks for state management
 * - Communicates with FastAPI backend on port 8000
 * - Handles two modes: Camera (real-time) and Upload (file-based)
 * - Displays recognition results with confidence scores
 * - Provides audio feedback via TextToSpeech component
 * 
 * Component Structure:
 * - App (this file) - Main container and state management
 * - CameraCapture - Real-time camera feed and detection
 * - ImageUpload - File upload interface
 * - ResultDisplay - Shows recognized text and confidence
 * - TextToSpeech - Audio output for recognized text
 * - VoiceGuidance - Audio instructions for blind users (in CameraCapture)
 */

import { useState, useRef } from 'react'
import './App.css'
import ImageUpload from './components/ImageUpload'
import CameraCapture from './components/CameraCapture'
import ResultDisplay from './components/ResultDisplay'
import TextToSpeech from './components/TextToSpeech'

function App() {
  // ==================== STATE MANAGEMENT ====================
  
  // Recognition results from backend
  const [recognizedText, setRecognizedText] = useState('')  // Recognized/corrected text
  const [confidence, setConfidence] = useState(0)            // Confidence score (0.0 to 1.0)
  
  // UI state
  const [loading, setLoading] = useState(false)              // Whether backend is processing
  const [error, setError] = useState('')                    // Error message to display
  const [uploadedImage, setUploadedImage] = useState(null)  // Preview image for upload mode
  const [activeTab, setActiveTab] = useState('camera')     // Current tab: 'camera' or 'upload'
  
  // Camera mode specific state
  const [lowConfidenceCount, setLowConfidenceCount] = useState(0)  // Track low confidence results
  const [guidance, setGuidance] = useState('')                    // Directional guidance from backend
  
  // Configuration
  const CONFIDENCE_THRESHOLD = 0.7  // Minimum confidence to show warning in camera mode

  /**
   * Handle image upload/processing from both camera and file upload.
   * 
   * This function:
   * 1. Prepares the image file for backend processing
   * 2. Sends it to the FastAPI backend for recognition
   * 3. Handles the response and updates UI state
   * 4. Manages different behavior for camera vs upload modes
   * 
   * @param {File} file - Image file to process (from camera capture or file upload)
   */
  const handleImageUpload = async (file) => {
    setLoading(true)  // Show loading indicator
    setError('')      // Clear any previous errors
    
    // Only clear previous results for upload mode (camera mode keeps previous results)
    if (activeTab === 'upload') {
      setRecognizedText('')
      setConfidence(0)
    }

    // Create image preview for upload mode only (camera shows live feed)
    if (activeTab === 'upload') {
      const reader = new FileReader()
      reader.onload = (e) => {
        setUploadedImage(e.target.result)  // Store base64 image for preview
      }
      reader.readAsDataURL(file)  // Convert file to base64 data URL
    }

    // Prepare form data for backend upload
    const formData = new FormData()
    formData.append('photo', file)  // Add image file to form

    // Determine if this is from camera or upload (affects backend processing)
    const isCamera = activeTab === 'camera'
    const apiUrl = `http://localhost:8000/api/recognize?use_enhancement=false&is_camera=${isCamera}`

    console.log('[APP] Sending request to backend:', {
      url: apiUrl,
      isCamera: isCamera,
      fileSize: file.size,
      fileName: file.name,
      fileType: file.type
    })

    try {
      const response = await fetch(apiUrl, {
        method: 'POST',
        body: formData,
      })

      console.log('[APP] Response received:', {
        status: response.status,
        statusText: response.statusText,
        ok: response.ok,
        headers: Object.fromEntries(response.headers.entries())
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }))
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`)
      }

      // Extract response data from backend
      const data = await response.json()
      const newText = data.word || ''              // Recognized/corrected text
      const newConfidence = data.confidence || 0    // Confidence score
      const guidance = data.guidance || null        // Directional guidance (camera mode only)
      
      // ========== CAMERA MODE PROCESSING ==========
      // Camera mode always shows results (even low confidence) for real-time feedback
      if (activeTab === 'camera') {
        console.log('[APP] Camera mode - received result:', {
          text: newText,
          confidence: newConfidence,
          guidance: guidance,
          braille_detected: data.braille_detected
        })
        
        // If text was recognized, update results
        if (newText && newText.trim() !== '') {
          // Only update if text changed (avoid unnecessary re-renders)
          if (newText !== recognizedText) {
            console.log('[APP] Updating camera result (always show):', newText, 'confidence:', newConfidence)
            setRecognizedText(newText)
            setConfidence(newConfidence)
            setLowConfidenceCount(0)  // Reset low confidence counter
          } else {
            console.log('[APP] Text unchanged, skipping update')
          }
        } else {
          // No text recognized - check if guidance was provided (no braille detected)
          console.log('[APP] Empty text received')
          if (guidance) {
            // Backend detected no braille and provided directional guidance
            console.log('[APP] Guidance received for no braille:', guidance)
            setGuidance(guidance)  // Store for VoiceGuidance component
            setRecognizedText('')   // Clear text
            setConfidence(0)
          } else {
            setGuidance('')  // Clear guidance if no new guidance
          }
        }
      } else {
        // ========== UPLOAD MODE PROCESSING ==========
        // Upload mode: always update results (no confidence threshold)
        if (newText !== recognizedText) {
          setRecognizedText(newText)
          setConfidence(newConfidence)
        }
      }
    } catch (err) {
      console.error('[APP] Recognition error details:', {
        name: err.name,
        message: err.message,
        stack: err.stack,
        type: typeof err
      })
      
      // Check if it's a network error
      if (err.name === 'TypeError' && err.message.includes('fetch')) {
        setError('Cannot connect to backend server. Please ensure the backend is running on http://localhost:8000')
      } else if (err.name === 'TypeError' && err.message.includes('Failed to fetch')) {
        setError('Network error: Cannot reach backend server. Check if backend is running.')
      } else {
        setError(err.message || 'Failed to recognize braille text. Please try again.')
      }
    } finally {
      setLoading(false)
    }
  }

  // ==================== RENDER ====================
  return (
    <div className="app">
      {/* Application header with title and description */}
      <header className="app-header">
        <h1>Braille Reader</h1>
        <p className="subtitle">Real-time Braille Recognition</p>
        <p className="description">Capture or upload braille text for instant recognition</p>
      </header>

      <main className="app-main">
        {/* Tab switcher: Camera mode vs Upload mode */}
        <div className="tab-container">
          <button 
            className={`tab-button ${activeTab === 'camera' ? 'active' : ''}`}
            onClick={() => setActiveTab('camera')}  // Switch to camera mode
          >
            📷 Camera
          </button>
          <button 
            className={`tab-button ${activeTab === 'upload' ? 'active' : ''}`}
            onClick={() => setActiveTab('upload')}  // Switch to upload mode
          >
            📁 Upload
          </button>
        </div>

        {/* Conditionally render Camera or Upload component based on active tab */}
        {activeTab === 'camera' ? (
          <div className="camera-section">
            <CameraCapture 
              onCapture={handleImageUpload}        // Callback when frame is captured
              loading={loading}                    // Loading state
              recognizedText={recognizedText}      // Current recognition result
              confidence={confidence}              // Confidence score
              guidance={guidance}                  // Directional guidance (if no braille detected)
            />
          </div>
        ) : (
          <div className="upload-section">
            <ImageUpload 
              onImageUpload={handleImageUpload}    // Callback when file is selected
              loading={loading}                    // Loading state
              uploadedImage={uploadedImage}        // Preview image (base64)
            />
          </div>
        )}

        {/* Error message display */}
        {error && (
          <div className="error-message">
            <span>⚠️</span> {error}
          </div>
        )}

        {/* Results section: only shown when text is recognized */}
        {recognizedText && (
          <div className="results-section">
            {/* Low confidence warning (camera mode only) */}
            {activeTab === 'camera' && confidence < CONFIDENCE_THRESHOLD && (
              <div className="low-confidence-warning">
                <span>⚠️</span> Low confidence result - adjusting camera position...
              </div>
            )}
            {/* Display recognized text and confidence bar */}
            <ResultDisplay 
              text={recognizedText} 
              confidence={confidence}
            />
            {/* Text-to-speech component with auto-play enabled */}
            <TextToSpeech text={recognizedText} autoPlay={true} />
          </div>
        )}
      </main>
    </div>
  )
}

export default App


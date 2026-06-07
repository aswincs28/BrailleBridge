/**
 * ResultDisplay Component
 * =======================
 * 
 * Displays the recognized braille text and confidence score.
 * 
 * Features:
 * - Large, readable text display
 * - Visual confidence bar (0-100%)
 * - Percentage confidence indicator
 * 
 * Props:
 * - text: Recognized text string
 * - confidence: Confidence score (0.0 to 1.0)
 */

import './ResultDisplay.css'

function ResultDisplay({ text, confidence }) {
  const confidencePercentage = Math.round(confidence * 100)

  return (
    <div className="result-display">
      <h2 className="result-title">Recognized Text</h2>
      <div className="result-text-container">
        <p className="result-text">{text || 'No text recognized'}</p>
      </div>
      {confidence > 0 && (
        <div className="confidence-indicator">
          <span className="confidence-label">Confidence:</span>
          <div className="confidence-bar-container">
            <div 
              className="confidence-bar" 
              style={{ width: `${confidencePercentage}%` }}
            ></div>
          </div>
          <span className="confidence-value">{confidencePercentage}%</span>
        </div>
      )}
    </div>
  )
}

export default ResultDisplay


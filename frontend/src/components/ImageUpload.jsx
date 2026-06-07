/**
 * ImageUpload Component
 * =====================
 * 
 * Provides a file upload interface for braille image recognition.
 * Supports both click-to-upload and drag-and-drop functionality.
 * 
 * Features:
 * - Drag and drop file upload
 * - Click to browse files
 * - Image preview after upload
 * - Loading state during processing
 * - File type validation (images only)
 * 
 * Props:
 * - onImageUpload: Callback function called with selected file
 * - loading: Whether backend is currently processing
 * - uploadedImage: Base64 data URL of uploaded image for preview
 */

import { useRef, useState } from 'react'
import './ImageUpload.css'

function ImageUpload({ onImageUpload, loading, uploadedImage }) {
  const fileInputRef = useRef(null)
  const [isDragging, setIsDragging] = useState(false)

  const handleFileSelect = (file) => {
    if (file && file.type.startsWith('image/')) {
      onImageUpload(file)
    } else {
      alert('Please select a valid image file')
    }
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setIsDragging(false)
    
    const file = e.dataTransfer.files[0]
    if (file) {
      handleFileSelect(file)
    }
  }

  const handleDragOver = (e) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = (e) => {
    e.preventDefault()
    setIsDragging(false)
  }

  const handleFileInputChange = (e) => {
    const file = e.target.files[0]
    if (file) {
      handleFileSelect(file)
    }
  }

  return (
    <div className="image-upload-container">
      <div
        className={`upload-area ${isDragging ? 'dragging' : ''} ${uploadedImage ? 'has-image' : ''}`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onClick={() => !loading && fileInputRef.current?.click()}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handleFileInputChange}
          style={{ display: 'none' }}
          disabled={loading}
        />

        {uploadedImage ? (
          <div className="image-preview">
            <img src={uploadedImage} alt="Uploaded braille" />
            <div className="preview-overlay">
              <span className="change-text">Click to change image</span>
            </div>
          </div>
        ) : (
          <div className="upload-placeholder">
            {loading ? (
              <>
                <div className="spinner"></div>
                <p>Processing braille image...</p>
              </>
            ) : (
              <>
                <div className="upload-icon">📷</div>
                <p className="upload-text">
                  <strong>Click to upload</strong> or drag and drop
                </p>
                <p className="upload-hint">Braille image (JPG, PNG, etc.)</p>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

export default ImageUpload


/**
 * CameraCapture Component
 * =======================
 * 
 * This component handles real-time camera capture and braille text recognition.
 * It provides:
 * - Live camera feed with detection box overlay
 * - Automatic frame capture and processing
 * - Image quality checks (brightness, contrast, blur, motion)
 * - Detection box visualization (center 60% width, 50% height)
 * - Integration with VoiceGuidance for audio instructions
 * - AI-powered instructions from backend
 * 
 * Key Features:
 * - Auto-detection: Captures frames every 1 second automatically
 * - Manual capture: User can trigger capture manually
 * - Stream management: Keeps camera stream alive with keep-alive mechanism
 * - Detection box: Visual guide showing where to position braille text
 * - Quality checks: Validates image quality before sending to backend
 * 
 * Props:
 * - onCapture: Callback function called with captured image file
 * - loading: Whether backend is currently processing
 * - recognizedText: Currently recognized text from backend
 * - confidence: Recognition confidence score
 * - guidance: Directional guidance from backend (when no braille detected)
 */

import { useState, useRef, useEffect } from 'react'
import './CameraCapture.css'
import VoiceGuidance from './VoiceGuidance'

function CameraCapture({ onCapture, loading, recognizedText, confidence, guidance }) {
  // ==================== REFS AND STATE ====================
  
  // DOM element references
  const videoRef = useRef(null)              // <video> element for camera feed
  const canvasRef = useRef(null)             // <canvas> element for frame capture
  const streamRef = useRef(null)             // MediaStream object from getUserMedia
  
  // Component state
  const [isActive, setIsActive] = useState(false)           // Whether camera is active
  const [error, setError] = useState('')                   // Error message
  const [isDetecting, setIsDetecting] = useState(false)    // Whether currently capturing frame
  const [isProcessing, setIsProcessing] = useState(false)  // Whether processing quality checks
  const [qualityStatus, setQualityStatus] = useState('')    // Status message for quality checks
  const [instructions, setInstructions] = useState('')      // AI-generated instructions from backend
  const [showInstructions, setShowInstructions] = useState(true)  // Whether to show instruction overlay
  const [currentImageFile, setCurrentImageFile] = useState(null)  // Current captured image for guidance
  
  // Interval/timeout references for cleanup
  const detectionIntervalRef = useRef(null)        // Auto-detection interval
  const keepAliveIntervalRef = useRef(null)        // Stream keep-alive interval
  const instructionCheckIntervalRef = useRef(null)  // Instruction update interval
  
  // Internal state tracking (using refs to avoid re-renders)
  const isMountedRef = useRef(true)                // Component mount status
  const shouldStopRef = useRef(false)              // Intentional stop flag
  const previousFrameRef = useRef(null)            // Previous frame for motion detection
  const lastProcessTimeRef = useRef(0)             // Last processing time
  const consecutiveFailuresRef = useRef(0)        // Track consecutive failures
  const currentIntervalRef = useRef(2000)            // Current capture interval (ms)
  const isCapturingRef = useRef(false)              // Lock to prevent overlapping captures
  const lastInstructionContextRef = useRef('')     // Last instruction context to avoid duplicates
  
  // Configuration constants for image quality checks
  const CONFIG = {
    PROCESSING_TIME: 400,        // Time (ms) for quality/stability analysis
    MIN_BRIGHTNESS: 20,           // Minimum average brightness (0-255) - lenient for camera
    MAX_BRIGHTNESS: 235,          // Maximum average brightness (0-255) - lenient for camera
    MIN_CONTRAST: 15,             // Minimum contrast (standard deviation) - lenient
    MIN_BLUR_THRESHOLD: 80,       // Laplacian variance threshold for blur detection - lenient
    MOTION_THRESHOLD: 0.15,       // Motion detection threshold (0-1), lower = more sensitive
    CONFIDENCE_THRESHOLD: 0.7,    // Minimum confidence to update results
    MIN_TIME_BETWEEN_UPDATES: 1000 // Minimum time (ms) between result updates
  }

  // Monitor confidence and recognition status to update instructions
  useEffect(() => {
    if (!isActive) return
    
    // Update instructions based on current recognition state
    if (confidence !== undefined) {
      if (confidence < 0.7 && confidence > 0 && lastInstructionContextRef.current !== 'low_confidence') {
        fetchInstructions('low_confidence')
      } else if ((!recognizedText || recognizedText.trim() === '') && lastInstructionContextRef.current !== 'no_text_detected') {
        // Only update to no_text_detected if we've been trying for a while
        const timeout = setTimeout(() => {
          if ((!recognizedText || recognizedText.trim() === '') && isActive) {
            fetchInstructions('no_text_detected')
          }
        }, 3000) // Wait 3 seconds before showing "no text detected" instructions
        return () => clearTimeout(timeout)
      }
    }
  }, [confidence, recognizedText, isActive])

  // Cleanup on unmount - only stop tracks, don't call stopCamera
  useEffect(() => {
    isMountedRef.current = true
    
    return () => {
      console.log('Component cleanup running...')
      isMountedRef.current = false
      
      // Clear instruction check interval
      if (instructionCheckIntervalRef.current) {
        clearInterval(instructionCheckIntervalRef.current)
        instructionCheckIntervalRef.current = null
      }
      
      // Only cleanup if we actually want to stop
      if (shouldStopRef.current || !streamRef.current) {
        console.log('Cleaning up streams...')
        stopKeepAlive()
        stopAutoDetection()
        if (streamRef.current) {
          streamRef.current.getTracks().forEach(track => {
            console.log('Stopping track on unmount:', track.label)
            track.stop()
          })
          streamRef.current = null
        }
        if (videoRef.current) {
          videoRef.current.srcObject = null
        }
      } else {
        console.log('Skipping cleanup - camera should stay active')
      }
    }
  }, [])

  // Fetch instructions from backend using Gemini
  const fetchInstructions = async (context = 'initial', voiceMode = false) => {
    try {
      // Don't fetch if we already have instructions for this context
      if (lastInstructionContextRef.current === context && instructions) {
        return
      }
      
      console.log('[CAMERA] Fetching instructions with context:', context, 'voiceMode:', voiceMode)
      const response = await fetch(`http://localhost:8000/api/camera-instructions?context=${context}&voice_mode=${voiceMode}`)
      
      if (response.ok) {
        const data = await response.json()
        setInstructions(data.instructions || '')
        lastInstructionContextRef.current = context
        console.log('[CAMERA] Instructions received:', data.instructions.substring(0, 50) + '...')
      } else {
        console.warn('[CAMERA] Failed to fetch instructions, using default')
        setInstructions('Position your braille text within the blue detection box. Ensure good lighting and keep the text steady.')
      }
    } catch (err) {
      console.error('[CAMERA] Error fetching instructions:', err)
      setInstructions('Position your braille text within the blue detection box. Ensure good lighting and keep the text steady.')
    }
  }

  const startCamera = async () => {
    try {
      setError('')
      shouldStopRef.current = false // Reset stop flag
      
      // Fetch initial instructions when camera starts
      await fetchInstructions('initial')
      setShowInstructions(true)
      
      // Request camera access
      const constraints = {
        video: {
          facingMode: 'environment', // Back camera on mobile
          width: { ideal: 1280 },
          height: { ideal: 720 }
        }
      }

      let stream = null
      try {
        stream = await navigator.mediaDevices.getUserMedia(constraints)
      } catch (err) {
        // Fallback: try without facingMode
        try {
          stream = await navigator.mediaDevices.getUserMedia({ video: true })
        } catch (err2) {
          throw new Error('Camera access denied or not available')
        }
      }

      // Video element should always be available now (always rendered)
      if (!videoRef.current) {
        console.error('Video element not available')
        stream.getTracks().forEach(track => track.stop())
        setError('Video element not ready. Please refresh the page.')
        return
      }
      
      console.log('Video element found:', videoRef.current)

      // Store stream BEFORE any async operations
      streamRef.current = stream
      const video = videoRef.current
      
      // Log stream info
      console.log('Stream obtained, tracks:', stream.getVideoTracks().length)
      stream.getVideoTracks().forEach((track, i) => {
        console.log(`Track ${i}:`, track.label, track.readyState, track.enabled)
        // Monitor track state changes
        track.onended = () => {
          console.error('Track ended unexpectedly!', track.label)
        }
        track.onmute = () => {
          console.warn('Track muted:', track.label)
        }
        track.onunmute = () => {
          console.log('Track unmuted:', track.label)
        }
      })

      // Set video properties BEFORE attaching stream
      video.muted = true
      video.playsInline = true
      video.setAttribute('playsinline', 'true')
      video.setAttribute('webkit-playsinline', 'true')
      video.setAttribute('autoplay', 'true')
      
      // Attach stream to video element
      video.srcObject = stream
      
      // Force video to be visible
      video.style.cssText = 'width: 100%; height: 100%; object-fit: cover; display: block;'

      // Wait for video to be ready
      await new Promise((resolve, reject) => {
        const timeout = setTimeout(() => {
          // Don't reject on timeout, just resolve to continue
          console.warn('Video metadata timeout, continuing anyway')
          cleanup()
          resolve()
        }, 3000)

        const onLoadedMetadata = () => {
          console.log('Video metadata loaded:', video.videoWidth, 'x', video.videoHeight)
          clearTimeout(timeout)
          cleanup()
          resolve()
        }

        const onCanPlay = () => {
          console.log('Video can play')
          clearTimeout(timeout)
          cleanup()
          resolve()
        }

        const onError = (e) => {
          clearTimeout(timeout)
          cleanup()
          console.error('Video error:', e, video.error)
          reject(new Error('Video failed to load: ' + (video.error?.message || 'Unknown')))
        }

        const cleanup = () => {
          video.removeEventListener('loadedmetadata', onLoadedMetadata)
          video.removeEventListener('canplay', onCanPlay)
          video.removeEventListener('error', onError)
        }

        // If video is already ready, resolve immediately
        if (video.readyState >= 2) {
          clearTimeout(timeout)
          cleanup()
          resolve()
          return
        }

        video.addEventListener('loadedmetadata', onLoadedMetadata, { once: true })
        video.addEventListener('canplay', onCanPlay, { once: true })
        video.addEventListener('error', onError, { once: true })

        // Try to play immediately - don't wait for it
        const playPromise = video.play()
        if (playPromise !== undefined) {
          playPromise.then(() => {
            console.log('Video play() succeeded immediately')
          }).catch(err => {
            console.warn('Video play() error (will retry):', err)
            // Don't reject, we'll retry later
          })
        }
      })

      // Ensure video is playing - try multiple times if needed
      let playAttempts = 0
      let videoPlaying = false
      
      while (video.paused && playAttempts < 5 && !videoPlaying) {
        try {
          await video.play()
          console.log('✓ Video play() succeeded on attempt', playAttempts + 1)
          videoPlaying = true
          break
        } catch (playErr) {
          playAttempts++
          console.warn('Video play() attempt', playAttempts, 'failed:', playErr.name, playErr.message)
          
          // Check if stream is still active
          const tracks = stream.getVideoTracks()
          const activeTracks = tracks.filter(t => t.readyState === 'live')
          console.log('Active tracks after play failure:', activeTracks.length, '/', tracks.length)
          
          if (playAttempts >= 5) {
            console.error('Failed to play video after 5 attempts, but keeping stream alive')
            // Don't throw - keep stream alive even if video doesn't play
          }
          await new Promise(resolve => setTimeout(resolve, 300))
        }
      }
      
      // Final check - ensure stream is still active
      const finalTracks = stream.getVideoTracks()
      const finalActiveTracks = finalTracks.filter(t => t.readyState === 'live')
      console.log('Final stream state - Active tracks:', finalActiveTracks.length, '/', finalTracks.length)
      
      if (finalActiveTracks.length === 0) {
        console.error('All tracks ended! This should not happen.')
        throw new Error('Camera stream ended unexpectedly')
      }

      // Verify stream is still active
      const activeTracks = stream.getVideoTracks().filter(track => track.readyState === 'live')
      if (activeTracks.length === 0) {
        throw new Error('Camera stream ended unexpectedly')
      }

      console.log('Camera active, tracks:', activeTracks.length)
      
      // Keep stream alive by periodically checking and restarting if needed
      startKeepAlive()
      
      setIsActive(true)
      
      // Start auto-detection immediately after camera is ready
      startAutoDetection()
      
      // Update instructions based on recognition results periodically
      // Clear any existing interval first
      if (instructionCheckIntervalRef.current) {
        clearInterval(instructionCheckIntervalRef.current)
      }
      
      instructionCheckIntervalRef.current = setInterval(() => {
        if (!isMountedRef.current || !isActive) {
          return
        }
        // Check if we need to update instructions based on current state
        if (confidence !== undefined && confidence < 0.7 && lastInstructionContextRef.current !== 'low_confidence') {
          fetchInstructions('low_confidence')
        } else if ((!recognizedText || recognizedText.trim() === '') && lastInstructionContextRef.current !== 'no_text_detected') {
          fetchInstructions('no_text_detected')
        }
      }, 5000) // Check every 5 seconds

    } catch (err) {
      console.error('Camera error:', err)
      let errorMsg = 'Unable to access camera. '
      if (err.name === 'NotAllowedError') {
        errorMsg += 'Please allow camera access in browser settings.'
      } else if (err.name === 'NotFoundError') {
        errorMsg += 'No camera found on this device.'
      } else if (err.name === 'NotReadableError') {
        errorMsg += 'Camera is being used by another application.'
      } else {
        errorMsg += err.message || 'Please check permissions.'
      }
      setError(errorMsg)
    }
  }

  const startKeepAlive = () => {
    stopKeepAlive() // Clear any existing interval
    
    keepAliveIntervalRef.current = setInterval(() => {
      if (streamRef.current && videoRef.current) {
        const tracks = streamRef.current.getVideoTracks()
        const activeTracks = tracks.filter(track => track.readyState === 'live')
        
        // If tracks are not live, try to restart video
        if (activeTracks.length === 0 && tracks.length > 0) {
          console.warn('Stream tracks not live, attempting to restart...')
          const video = videoRef.current
          if (video.paused) {
            video.play().catch(err => {
              console.warn('Keep-alive play failed:', err)
            })
          }
        }
        
        // Ensure video is playing
        if (videoRef.current && videoRef.current.paused) {
          videoRef.current.play().catch(() => {
            // Ignore play errors in keep-alive
          })
        }
      }
    }, 1000) // Check every second
  }

  const stopKeepAlive = () => {
    if (keepAliveIntervalRef.current) {
      clearInterval(keepAliveIntervalRef.current)
      keepAliveIntervalRef.current = null
    }
  }

  const stopCamera = () => {
    // Prevent multiple calls
    if (shouldStopRef.current && !streamRef.current) {
      console.log('stopCamera() already called, skipping')
      return
    }
    
    console.log('stopCamera() called - intentional stop')
    
    // Mark that we intentionally want to stop
    shouldStopRef.current = true
    
    // Stop keep-alive
    stopKeepAlive()
    
    // Stop auto-detection first
    stopAutoDetection()
    
    // Clear instruction check interval
    if (instructionCheckIntervalRef.current) {
      clearInterval(instructionCheckIntervalRef.current)
      instructionCheckIntervalRef.current = null
    }

    // Stop all tracks
    if (streamRef.current) {
      const tracks = streamRef.current.getTracks()
      console.log('Stopping', tracks.length, 'tracks')
      tracks.forEach(track => {
        console.log('Stopping track:', track.kind, track.readyState, track.label)
        track.stop()
      })
      streamRef.current = null
    }

    // Clear video source
    if (videoRef.current) {
      videoRef.current.pause()
      videoRef.current.srcObject = null
    }

    setIsActive(false)
    console.log('Camera stopped')
  }

  const startAutoDetection = () => {
    stopAutoDetection() // Clear any existing interval

    const captureLoop = async () => {
      // Only proceed if no processing is happening
      if (!loading && !isDetecting && !isProcessing && videoRef.current && canvasRef.current) {
        const video = videoRef.current
        // Only capture if video is ready
        if (video.readyState >= 2 && video.videoWidth > 0 && video.videoHeight > 0) {
          // Wait for captureFrame to complete (including backend processing + Gemini correction)
          await captureFrame()
        }
      }
      
      // Schedule next capture ONLY after current one is completely done
      // Use a fixed delay between captures
      const delay = 1000 // 1 second between captures
      detectionIntervalRef.current = setTimeout(captureLoop, delay)
    }

    // Start the loop
    captureLoop()
  }

  const stopAutoDetection = () => {
    if (detectionIntervalRef.current) {
      clearTimeout(detectionIntervalRef.current)
      detectionIntervalRef.current = null
    }
  }

  // Image quality check functions
  const checkImageQuality = (canvas) => {
    const ctx = canvas.getContext('2d')
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height)
    const data = imageData.data
    const pixelCount = canvas.width * canvas.height

    // Calculate brightness (average grayscale value)
    let brightnessSum = 0
    for (let i = 0; i < data.length; i += 4) {
      // Grayscale: 0.299*R + 0.587*G + 0.114*B
      const gray = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]
      brightnessSum += gray
    }
    const avgBrightness = brightnessSum / pixelCount

    // Calculate contrast (standard deviation of brightness)
    let varianceSum = 0
    for (let i = 0; i < data.length; i += 4) {
      const gray = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]
      varianceSum += Math.pow(gray - avgBrightness, 2)
    }
    const contrast = Math.sqrt(varianceSum / pixelCount)

    // Calculate blur (Laplacian variance - simple edge detection)
    let laplacianSum = 0
    const laplacianKernel = [
      [0, -1, 0],
      [-1, 4, -1],
      [0, -1, 0]
    ]
    
    // Sample a smaller area for performance (every 4th pixel)
    const step = 4
    for (let y = 1; y < canvas.height - 1; y += step) {
      for (let x = 1; x < canvas.width - 1; x += step) {
        let laplacianValue = 0
        for (let ky = -1; ky <= 1; ky++) {
          for (let kx = -1; kx <= 1; kx++) {
            const idx = ((y + ky) * canvas.width + (x + kx)) * 4
            const gray = 0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2]
            laplacianValue += gray * laplacianKernel[ky + 1][kx + 1]
          }
        }
        laplacianSum += Math.abs(laplacianValue)
      }
    }
    const blurVariance = laplacianSum / ((canvas.width / step) * (canvas.height / step))

    // Quality checks
    const isBrightnessOK = avgBrightness >= CONFIG.MIN_BRIGHTNESS && avgBrightness <= CONFIG.MAX_BRIGHTNESS
    const isContrastOK = contrast >= CONFIG.MIN_CONTRAST
    const isBlurOK = blurVariance >= CONFIG.MIN_BLUR_THRESHOLD

    return {
      passed: isBrightnessOK && isContrastOK && isBlurOK,
      brightness: avgBrightness,
      contrast: contrast,
      blur: blurVariance,
      details: {
        brightnessOK: isBrightnessOK,
        contrastOK: isContrastOK,
        blurOK: isBlurOK
      }
    }
  }

  // Frame comparison for motion detection
  const compareFrames = (canvas1, canvas2) => {
    if (!canvas1 || !canvas2) return { motion: 1.0, passed: false }

    const ctx1 = canvas1.getContext('2d')
    const ctx2 = canvas2.getContext('2d')
    const imageData1 = ctx1.getImageData(0, 0, canvas1.width, canvas1.height)
    const imageData2 = ctx2.getImageData(0, 0, canvas2.width, canvas2.height)
    const data1 = imageData1.data
    const data2 = imageData2.data

    // Calculate histogram difference (simpler and faster than pixel-by-pixel)
    const hist1 = new Array(256).fill(0)
    const hist2 = new Array(256).fill(0)

    // Sample pixels for performance (every 8th pixel)
    const step = 8
    for (let i = 0; i < data1.length; i += 4 * step) {
      const gray1 = Math.round(0.299 * data1[i] + 0.587 * data1[i + 1] + 0.114 * data1[i + 2])
      const gray2 = Math.round(0.299 * data2[i] + 0.587 * data2[i + 1] + 0.114 * data2[i + 2])
      hist1[Math.min(255, Math.max(0, gray1))]++
      hist2[Math.min(255, Math.max(0, gray2))]++
    }

    // Calculate histogram difference
    let diffSum = 0
    let totalPixels = 0
    for (let i = 0; i < 256; i++) {
      diffSum += Math.abs(hist1[i] - hist2[i])
      totalPixels += hist1[i] + hist2[i]
    }
    const motionRatio = diffSum / totalPixels

    return {
      motion: motionRatio,
      passed: motionRatio < CONFIG.MOTION_THRESHOLD
    }
  }

  /**
   * Capture a frame from the video stream and send it to backend for recognition.
   * 
   * This function:
   * 1. Captures current video frame to canvas
   * 2. Extracts the detection box region (center 60% width, 50% height)
   * 3. Converts to JPEG blob
   * 4. Sends to backend via onCapture callback
   * 5. Stores image for directional guidance
   * 
   * The detection box is a smaller region in the center of the frame where
   * users should position their braille text. This helps focus recognition
   * on the relevant area and reduces background noise.
   */
  const captureFrame = async () => {
    const video = videoRef.current
    const canvas = canvasRef.current

    // Prevent overlapping captures - if already capturing, skip this frame
    if (isCapturingRef.current) {
      console.log('[CAMERA] Capture already in progress, skipping')
      return
    }

    // Validate prerequisites
    if (!video || !canvas || loading || isDetecting || isProcessing) {
      return
    }

    // Check video is ready (must be loaded and have valid dimensions)
    if (video.readyState < 2 || video.videoWidth === 0 || video.videoHeight === 0) {
      return
    }

    // Set capture lock to prevent overlapping captures
    isCapturingRef.current = true

    try {
      setIsDetecting(true)  // Show detecting indicator
      setQualityStatus('Capturing...')

      // Set canvas to match video dimensions
      canvas.width = video.videoWidth
      canvas.height = video.videoHeight
      const ctx = canvas.getContext('2d')

      // Draw current video frame to canvas
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height)

      // ========== EXTRACT DETECTION BOX REGION ==========
      // The detection box is centered and smaller than full frame:
      // - Width: 60% of frame width
      // - Height: 50% of frame height
      // This focuses recognition on the center area where users position braille
      const boxWidth = canvas.width * 0.6
      const boxHeight = canvas.height * 0.5
      const boxX = (canvas.width - boxWidth) / 2  // Center horizontally
      const boxY = (canvas.height - boxHeight) / 2  // Center vertically

      // Create new canvas for cropped detection box
      const detectionBoxCanvas = document.createElement('canvas')
      detectionBoxCanvas.width = boxWidth
      detectionBoxCanvas.height = boxHeight
      const detectionBoxCtx = detectionBoxCanvas.getContext('2d')

      // Copy detection box region from main canvas to cropped canvas
      detectionBoxCtx.drawImage(
        canvas,
        boxX, boxY, boxWidth, boxHeight,  // Source region
        0, 0, boxWidth, boxHeight           // Destination (full cropped canvas)
      )

      // Convert cropped canvas to JPEG blob for backend upload
      console.log('[CAMERA] Sending detection box frame to backend, size:', detectionBoxCanvas.width, 'x', detectionBoxCanvas.height)
      
      // Convert toBlob to Promise for async/await
      const blobPromise = new Promise((resolve, reject) => {
        detectionBoxCanvas.toBlob((blob) => {
          if (blob && blob.size > 0) {
            resolve(blob)
          } else {
            reject(new Error('Failed to create blob from canvas'))
          }
        }, 'image/jpeg', 0.9)  // 90% quality JPEG
      })

      const blob = await blobPromise
      console.log('[CAMERA] Frame blob created, size:', blob.size, 'bytes')
      setQualityStatus('Processing...')
      
      // Create File object from blob
      const file = new File([blob], 'capture.jpg', { type: 'image/jpeg' })
      
      // Store current image for directional guidance component
      setCurrentImageFile(file)
      
      console.log('[CAMERA] Calling onCapture with file - waiting for backend processing...')
      console.log('[CAMERA] File details:', {
        name: file.name,
        size: file.size,
        type: file.type
      })
      
      // Send to backend and wait for processing to complete
      // Backend will: auto-crop, preprocess, run model, correct with Gemini
      await onCapture(file)
      console.log('[CAMERA] Backend processing completed - ready for next capture')
      
    } catch (err) {
      console.error('[CAMERA] Error during capture:', err)
      setError('Capture failed: ' + err.message)
    } finally {
      setIsDetecting(false)
      setQualityStatus('')
      // Release capture lock
      isCapturingRef.current = false
    }
  }

  const handleManualCapture = () => {
    captureFrame()
  }

  return (
    <div className="camera-capture">
      <VoiceGuidance
        isActive={isActive}
        confidence={confidence}
        recognizedText={recognizedText}
        isDetecting={isDetecting || isProcessing}
        instructions={instructions}
        onFetchInstructions={fetchInstructions}
        currentImageFile={currentImageFile}
        guidance={guidance}
      />
      <div className="camera-container">
        {/* Always render video element (hidden when not active) */}
        <div className="video-wrapper" style={{ display: isActive ? 'block' : 'none' }}>
          <video
            ref={videoRef}
            className="camera-video"
            autoPlay
            playsInline
            muted
          />
          {isActive && (
            <>
              <div className="detection-box">
                <div className="detection-box-border"></div>
                <div className="detection-box-label">Position braille here</div>
              </div>
              {showInstructions && instructions && (
                <div className="camera-instructions">
                  <div className="instructions-content">
                    <span className="instructions-icon">💡</span>
                    <p>{instructions}</p>
                    <button 
                      className="instructions-close" 
                      onClick={() => setShowInstructions(false)}
                      aria-label="Close instructions"
                    >
                      ×
                    </button>
                  </div>
                </div>
              )}
              {(isDetecting || isProcessing) && (
                <div className="detecting-overlay">
                  <div className="detecting-spinner"></div>
                  <p>{isProcessing ? qualityStatus || 'Analyzing frame...' : 'Detecting braille...'}</p>
                </div>
              )}
            </>
          )}
        </div>

        {!isActive && (
          <div className="camera-placeholder">
            <div className="camera-icon">📷</div>
            <p>Camera not active</p>
            <button className="camera-button start-button" onClick={startCamera}>
              Start Camera
            </button>
          </div>
        )}

        {isActive && (
          <div className="camera-controls">
            {!showInstructions && (
              <button 
                className="camera-button instructions-button" 
                onClick={() => {
                  setShowInstructions(true)
                  fetchInstructions('initial')
                }}
                title="Show instructions"
              >
                💡 Show Instructions
              </button>
            )}
            <button 
              className="camera-button capture-button" 
              onClick={handleManualCapture} 
              disabled={loading || isDetecting || isProcessing}
            >
              📸 Capture Now
            </button>
            <button className="camera-button stop-button" onClick={stopCamera}>
              ⏹️ Stop Camera
            </button>
          </div>
        )}
        <canvas ref={canvasRef} style={{ display: 'none' }} />
      </div>
      {error && (
        <div className="camera-error">
          <span>⚠️</span> {error}
        </div>
      )}
    </div>
  )
}

export default CameraCapture

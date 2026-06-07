import { useEffect, useRef, useState } from 'react'
import './VoiceGuidance.css'

/**
 * VoiceGuidance Component
 * ======================
 * 
 * Provides audio instructions and feedback to help blind users position
 * braille text correctly in the camera detection box.
 * 
 * Features:
 * - Initial "Camera is ready" message (once per session)
 * - Real-time feedback based on confidence and recognition results
 * - AI-powered directional guidance (move left, right, closer, etc.)
 * - Audio cues (beeps) for different feedback types
 * - Queued speech to prevent overlapping audio
 * - Periodic guidance reminders
 * - Integration with backend Gemini API for contextual instructions
 * 
 * Audio Feedback Types:
 * - Success beep: High confidence recognition
 * - Warning beep: Low confidence or no text detected
 * - Info beep: Detection in progress
 * - Error beep: Critical issues
 * 
 * Props:
 * - isActive: Whether camera is currently active
 * - confidence: Current recognition confidence (0.0 to 1.0)
 * - recognizedText: Currently recognized text
 * - isDetecting: Whether frame is currently being processed
 * - instructions: AI-generated instructions from backend
 * - onFetchInstructions: Callback to fetch new instructions
 * - currentImageFile: Current captured image for directional guidance
 * - guidance: Directional guidance from backend (when no braille detected)
 */
function VoiceGuidance({ 
  isActive, 
  confidence, 
  recognizedText, 
  isDetecting,
  instructions,
  onFetchInstructions,
  currentImageFile,
  guidance
}) {
  const synthRef = useRef(null)
  const utteranceRef = useRef(null)
  const lastConfidenceRef = useRef(null)
  const lastTextRef = useRef('')
  const guidanceQueueRef = useRef([])
  const isSpeakingRef = useRef(false)
  const [voiceEnabled, setVoiceEnabled] = useState(true)
  const [selectedVoice, setSelectedVoice] = useState(null)
  const consecutiveLowConfidenceRef = useRef(0)
  const noDetectionCountRef = useRef(0)
  const cameraReadySaidRef = useRef(false)
  const lastIsActiveRef = useRef(false)
  const lastGuidanceTimeRef = useRef(0)
  const guidanceCooldownRef = useRef(3000) // 3 seconds between guidance updates
  const lastGuidanceRef = useRef('')

  // Initialize speech synthesis
  useEffect(() => {
    const loadVoices = () => {
      const voices = window.speechSynthesis.getVoices()
      if (voices.length > 0 && !selectedVoice) {
        // Prefer a clear, natural English voice
        const preferredVoice = voices.find(
          voice => voice.lang.startsWith('en') && 
          (voice.name.includes('Natural') || voice.name.includes('Enhanced') || voice.name.includes('Premium'))
        ) || voices.find(voice => voice.lang.startsWith('en')) || voices[0]
        setSelectedVoice(preferredVoice)
      }
    }
    
    loadVoices()
    window.speechSynthesis.onvoiceschanged = loadVoices
    
    return () => {
      if (window.speechSynthesis) {
        window.speechSynthesis.cancel()
      }
    }
  }, [selectedVoice])

  // Speak text with queuing to prevent overlap
  const speak = (text, priority = 'normal', interrupt = false) => {
    if (!voiceEnabled || !text || text.trim() === '') return

    // Cancel current speech if interrupt is true
    if (interrupt && window.speechSynthesis.speaking) {
      window.speechSynthesis.cancel()
      isSpeakingRef.current = false
    }

    // If already speaking and not interrupting, queue the message
    if (window.speechSynthesis.speaking && !interrupt) {
      guidanceQueueRef.current.push({ text, priority })
      return
    }

    const utterance = new SpeechSynthesisUtterance(text)
    utterance.voice = selectedVoice
    utterance.rate = 1.0 // Slightly slower for clarity
    utterance.pitch = 1.0
    utterance.volume = 1.0

    utterance.onstart = () => {
      isSpeakingRef.current = true
    }

    utterance.onend = () => {
      isSpeakingRef.current = false
      // Process next item in queue
      if (guidanceQueueRef.current.length > 0) {
        const next = guidanceQueueRef.current.shift()
        setTimeout(() => speak(next.text, next.priority, false), 300)
      }
    }

    utterance.onerror = (e) => {
      console.error('Voice guidance error:', e)
      isSpeakingRef.current = false
    }

    utteranceRef.current = utterance
    window.speechSynthesis.speak(utterance)
  }

  // Play audio cue (beep) for quick feedback
  const playBeep = (type = 'info') => {
    if (!voiceEnabled) return
    
    const audioContext = new (window.AudioContext || window.webkitAudioContext)()
    const oscillator = audioContext.createOscillator()
    const gainNode = audioContext.createGain()

    oscillator.connect(gainNode)
    gainNode.connect(audioContext.destination)

    // Different beep patterns for different feedback types
    switch (type) {
      case 'success':
        // High-pitched double beep
        oscillator.frequency.value = 800
        gainNode.gain.setValueAtTime(0.3, audioContext.currentTime)
        gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.1)
        oscillator.start(audioContext.currentTime)
        oscillator.stop(audioContext.currentTime + 0.1)
        setTimeout(() => {
          const osc2 = audioContext.createOscillator()
          const gain2 = audioContext.createGain()
          osc2.connect(gain2)
          gain2.connect(audioContext.destination)
          osc2.frequency.value = 1000
          gain2.gain.setValueAtTime(0.3, audioContext.currentTime)
          gain2.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.1)
          osc2.start(audioContext.currentTime)
          osc2.stop(audioContext.currentTime + 0.1)
        }, 150)
        break
      case 'warning':
        // Medium-pitched beep
        oscillator.frequency.value = 600
        gainNode.gain.setValueAtTime(0.3, audioContext.currentTime)
        gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.2)
        oscillator.start(audioContext.currentTime)
        oscillator.stop(audioContext.currentTime + 0.2)
        break
      case 'error':
        // Low-pitched beep
        oscillator.frequency.value = 400
        gainNode.gain.setValueAtTime(0.3, audioContext.currentTime)
        gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.3)
        oscillator.start(audioContext.currentTime)
        oscillator.stop(audioContext.currentTime + 0.3)
        break
      default:
        // Short info beep
        oscillator.frequency.value = 500
        gainNode.gain.setValueAtTime(0.2, audioContext.currentTime)
        gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.1)
        oscillator.start(audioContext.currentTime)
        oscillator.stop(audioContext.currentTime + 0.1)
    }
  }

  // Fetch AI-powered directional guidance
  const fetchDirectionalGuidance = async () => {
    if (!currentImageFile || !voiceEnabled) return
    
    const now = Date.now()
    if (now - lastGuidanceTimeRef.current < guidanceCooldownRef.current) {
      return // Cooldown to avoid too frequent API calls
    }
    
    lastGuidanceTimeRef.current = now
    
    try {
      const formData = new FormData()
      formData.append('photo', currentImageFile)
      
      const response = await fetch(
        `http://localhost:8000/api/directional-guidance?confidence=${confidence || 0}&recognized_text=${encodeURIComponent(recognizedText || '')}`,
        {
          method: 'POST',
          body: formData
        }
      )
      
      if (response.ok) {
        const data = await response.json()
        if (data.guidance) {
          speak(data.guidance, 'normal', false)
        }
      }
    } catch (err) {
      console.error('[VOICE] Error fetching directional guidance:', err)
      // Fallback to generic guidance
      if (confidence < 0.6) {
        speak('Try adjusting the camera position', 'normal', false)
      }
    }
  }

  // Fetch voice-optimized instructions when camera starts
  useEffect(() => {
    if (isActive && onFetchInstructions) {
      // Fetch voice-optimized instructions
      onFetchInstructions('initial', true)
    }
  }, [isActive, onFetchInstructions])

  // Reset camera ready flag when camera turns off
  useEffect(() => {
    if (lastIsActiveRef.current && !isActive) {
      // Camera was turned off, reset the flag
      cameraReadySaidRef.current = false
      noDetectionCountRef.current = 0
      consecutiveLowConfidenceRef.current = 0
      lastTextRef.current = ''
      lastConfidenceRef.current = null
    }
    lastIsActiveRef.current = isActive
  }, [isActive])

  // Initial instructions when camera starts (only once per session)
  useEffect(() => {
    if (isActive && !cameraReadySaidRef.current) {
      // Wait a moment for camera to stabilize
      const timeout = setTimeout(() => {
        if (!cameraReadySaidRef.current) {
          cameraReadySaidRef.current = true
          if (instructions) {
            const initialGuidance = `Camera is ready. ${instructions} Position your braille text within the detection box in the center of the screen. Keep the text steady and ensure good lighting.`
            speak(initialGuidance, 'high', true)
          } else {
            // Fallback instructions if Gemini instructions not loaded yet
            const initialGuidance = `Camera is ready. Position your braille text within the blue detection box in the center of the screen. Keep the text steady and ensure good lighting. Move closer if needed.`
            speak(initialGuidance, 'high', true)
          }
        }
      }, 2000) // Wait a bit longer for instructions to load
      
      return () => clearTimeout(timeout)
    }
  }, [isActive, instructions])

  // Audio feedback when detection starts/stops
  useEffect(() => {
    if (!isActive) return
    
    if (isDetecting) {
      // Quick beep when detection starts
      playBeep('info')
    }
  }, [isDetecting, isActive])

  // Provide feedback based on confidence level
  useEffect(() => {
    if (!isActive || confidence === undefined || isDetecting) return

    // Text successfully detected
    if (recognizedText && recognizedText.trim() !== '') {
      if (recognizedText !== lastTextRef.current) {
        lastTextRef.current = recognizedText
        noDetectionCountRef.current = 0
        
        if (confidence >= 0.8) {
          // High confidence - success
          playBeep('success')
          speak(`Text detected: ${recognizedText}. Position is good.`, 'high', false)
        } else if (confidence >= 0.6) {
          // Medium confidence
          playBeep('info')
          speak(`Text detected: ${recognizedText}.`, 'normal', false)
        } else {
          // Low confidence - provide AI-powered guidance
          consecutiveLowConfidenceRef.current++
          playBeep('warning')
          
          // Get AI-powered directional guidance
          if (consecutiveLowConfidenceRef.current === 1 || consecutiveLowConfidenceRef.current === 3 || consecutiveLowConfidenceRef.current === 5) {
            fetchDirectionalGuidance()
          }
          
          if (consecutiveLowConfidenceRef.current === 5) {
            consecutiveLowConfidenceRef.current = 0 // Reset to avoid spamming
          }
        }
      }
    } else {
      // No text detected
      if (lastTextRef.current !== '') {
        lastTextRef.current = ''
        consecutiveLowConfidenceRef.current = 0
      }
      
      noDetectionCountRef.current++
      
      // Provide guidance after several failed attempts - use AI directional guidance
      // (Backend guidance is handled separately in useEffect)
      if (noDetectionCountRef.current === 5 || noDetectionCountRef.current === 10 || noDetectionCountRef.current === 15) {
        playBeep('warning')
        fetchDirectionalGuidance()
        if (noDetectionCountRef.current >= 10) {
          noDetectionCountRef.current = 0 // Reset to avoid spamming
        }
      }
    }

    // Track confidence changes for guidance
    if (lastConfidenceRef.current !== null) {
      const confidenceDiff = confidence - lastConfidenceRef.current
      
      // Significant improvement
      if (confidenceDiff > 0.2 && confidence >= 0.7) {
        playBeep('success')
        speak(`Position improved.`, 'normal', false)
      }
      // Significant degradation
      else if (confidenceDiff < -0.2 && confidence < 0.6) {
        playBeep('warning')
        fetchDirectionalGuidance()
      }
    }
    
    lastConfidenceRef.current = confidence
  }, [confidence, recognizedText, isActive, isDetecting, currentImageFile, voiceEnabled, guidance])
  
  // Handle guidance from backend when no braille is detected (Gemini pre-check)
  useEffect(() => {
    if (!isActive || isDetecting) return
    
    // If guidance is provided and different from last, speak it
    if (guidance && guidance.trim() !== '' && guidance !== lastGuidanceRef.current) {
      console.log('[VOICE] Speaking guidance from backend (no braille detected):', guidance)
      playBeep('warning')
      speak(guidance, 'normal', false)
      lastGuidanceRef.current = guidance
      // Reset detection counter since we provided guidance
      noDetectionCountRef.current = 0
    }
  }, [guidance, isActive, isDetecting])

  // Provide periodic guidance reminders with AI
  useEffect(() => {
    if (!isActive) return

    const guidanceInterval = setInterval(() => {
      if (!isSpeakingRef.current && (confidence < 0.7 || !recognizedText)) {
        // Use AI directional guidance instead of generic tips
        fetchDirectionalGuidance()
      }
    }, 15000) // Every 15 seconds

    return () => clearInterval(guidanceInterval)
  }, [isActive, confidence, recognizedText, currentImageFile, voiceEnabled])

  return (
    <div className="voice-guidance-controls">
      <button
        className={`voice-toggle ${voiceEnabled ? 'enabled' : 'disabled'}`}
        onClick={() => {
          setVoiceEnabled(!voiceEnabled)
          if (!voiceEnabled) {
            speak('Voice guidance enabled', 'high', true)
          } else {
            window.speechSynthesis.cancel()
            playBeep('info')
          }
        }}
        aria-label={voiceEnabled ? 'Disable voice guidance' : 'Enable voice guidance'}
        title={voiceEnabled ? 'Voice guidance: ON' : 'Voice guidance: OFF'}
      >
        {voiceEnabled ? '🔊 Voice ON' : '🔇 Voice OFF'}
      </button>
    </div>
  )
}

export default VoiceGuidance


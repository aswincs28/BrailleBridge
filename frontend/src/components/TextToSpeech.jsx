/**
 * TextToSpeech Component
 * ======================
 * 
 * Provides text-to-speech functionality using the Web Speech API.
 * Converts recognized braille text to audio output.
 * 
 * Features:
 * - Automatic playback when text changes (if autoPlay enabled)
 * - Manual play/pause/stop controls
 * - Voice selection (uses system voices)
 * - Adjustable rate, pitch, and volume
 * - Prevents duplicate playback of same text
 * 
 * Props:
 * - text: Text string to speak
 * - autoPlay: If true, automatically speaks when text changes
 */

import { useState, useEffect, useRef } from 'react'
import './TextToSpeech.css'

function TextToSpeech({ text, autoPlay = false }) {
  const [isPlaying, setIsPlaying] = useState(false)
  const [isPaused, setIsPaused] = useState(false)
  const [voices, setVoices] = useState([])
  const [selectedVoice, setSelectedVoice] = useState(null)
  const [rate, setRate] = useState(1)
  const [pitch, setPitch] = useState(1)
  const [volume, setVolume] = useState(1)
  const synthRef = useRef(null)
  const utteranceRef = useRef(null)
  const lastTextRef = useRef('')

  useEffect(() => {
    // Load available voices
    const loadVoices = () => {
      const availableVoices = window.speechSynthesis.getVoices()
      setVoices(availableVoices)
      
      // Try to set a default English voice
      if (availableVoices.length > 0 && !selectedVoice) {
        const englishVoice = availableVoices.find(
          voice => voice.lang.startsWith('en')
        ) || availableVoices[0]
        setSelectedVoice(englishVoice)
      }
    }

    loadVoices()
    window.speechSynthesis.onvoiceschanged = loadVoices

    return () => {
      if (utteranceRef.current) {
        window.speechSynthesis.cancel()
      }
    }
  }, [])

  // Auto-play when text changes (only if autoPlay is enabled and text is different)
  useEffect(() => {
    if (autoPlay && text && text.trim() !== '' && text !== lastTextRef.current) {
      lastTextRef.current = text
      // Cancel any ongoing speech first
      window.speechSynthesis.cancel()
      // Small delay to ensure previous speech is cancelled and text is ready
      const timeoutId = setTimeout(() => {
        speak()
      }, 100)
      
      return () => clearTimeout(timeoutId)
    }
  }, [text, autoPlay])


  const speak = () => {
    if (!text || text.trim() === '') {
      alert('No text to speak')
      return
    }

    // Cancel any ongoing speech
    window.speechSynthesis.cancel()

    const utterance = new SpeechSynthesisUtterance(text)
    utterance.voice = selectedVoice
    utterance.rate = rate
    utterance.pitch = pitch
    utterance.volume = volume

    utterance.onstart = () => {
      setIsPlaying(true)
      setIsPaused(false)
    }

    utterance.onend = () => {
      setIsPlaying(false)
      setIsPaused(false)
    }

    utterance.onerror = (e) => {
      console.error('Speech synthesis error:', e)
      setIsPlaying(false)
      setIsPaused(false)
    }

    utteranceRef.current = utterance
    window.speechSynthesis.speak(utterance)
  }

  const pauseSpeech = () => {
    if (window.speechSynthesis.speaking && !window.speechSynthesis.paused) {
      window.speechSynthesis.pause()
      setIsPaused(true)
    }
  }

  const resumeSpeech = () => {
    if (window.speechSynthesis.paused) {
      window.speechSynthesis.resume()
      setIsPaused(false)
    }
  }

  const stopSpeech = () => {
    window.speechSynthesis.cancel()
    setIsPlaying(false)
    setIsPaused(false)
  }

  if (!text || text.trim() === '') {
    return null
  }

  return (
    <div className="text-to-speech">
      <h3 className="tts-title">🔊 Voice Output</h3>
      
      <div className="tts-controls">
        <div className="tts-buttons">
          {/* Always show replay button */}
          <button 
            className="tts-button play-button"
            onClick={speak}
            aria-label="Replay speech"
            title="Replay"
          >
            🔄 Replay
          </button>
          
          {isPlaying && !isPaused && (
            <button 
              className="tts-button pause-button"
              onClick={pauseSpeech}
              aria-label="Pause speech"
            >
              ⏸️ Pause
            </button>
          )}
          
          {isPaused && (
            <button 
              className="tts-button resume-button"
              onClick={resumeSpeech}
              aria-label="Resume speech"
            >
              ▶️ Resume
            </button>
          )}
          
          {(isPlaying || isPaused) && (
            <button 
              className="tts-button stop-button"
              onClick={stopSpeech}
              aria-label="Stop speech"
            >
              ⏹️ Stop
            </button>
          )}
        </div>

        <div className="tts-settings">
          <div className="tts-setting">
            <label htmlFor="voice-select">Voice:</label>
            <select
              id="voice-select"
              value={selectedVoice ? selectedVoice.name : ''}
              onChange={(e) => {
                const voice = voices.find(v => v.name === e.target.value)
                setSelectedVoice(voice)
              }}
              disabled={isPlaying || isPaused}
            >
              {voices.map((voice, index) => (
                <option key={index} value={voice.name}>
                  {voice.name} ({voice.lang})
                </option>
              ))}
            </select>
          </div>

          <div className="tts-setting">
            <label htmlFor="rate-slider">Speed: {rate.toFixed(1)}x</label>
            <input
              id="rate-slider"
              type="range"
              min="0.5"
              max="2"
              step="0.1"
              value={rate}
              onChange={(e) => setRate(parseFloat(e.target.value))}
              disabled={isPlaying || isPaused}
            />
          </div>

          <div className="tts-setting">
            <label htmlFor="pitch-slider">Pitch: {pitch.toFixed(1)}</label>
            <input
              id="pitch-slider"
              type="range"
              min="0.5"
              max="2"
              step="0.1"
              value={pitch}
              onChange={(e) => setPitch(parseFloat(e.target.value))}
              disabled={isPlaying || isPaused}
            />
          </div>

          <div className="tts-setting">
            <label htmlFor="volume-slider">Volume: {Math.round(volume * 100)}%</label>
            <input
              id="volume-slider"
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={volume}
              onChange={(e) => setVolume(parseFloat(e.target.value))}
              disabled={isPlaying || isPaused}
            />
          </div>
        </div>
      </div>
    </div>
  )
}

export default TextToSpeech


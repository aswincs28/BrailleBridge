# Braille Recognition System

A real-time Braille text recognition system with camera capture and image upload capabilities. The system uses a CRNN (Convolutional Recurrent Neural Network) model for recognition and integrates with Google Gemini API for word correction.

## 🎯 Features

- **Real-time Camera Capture**: Capture braille text using device camera with automatic detection
- **Image Upload**: Upload and process braille images
- **Automatic Cropping**: Intelligent detection and cropping of braille regions
- **Word Correction**: AI-powered word correction using Google Gemini Flash 2.5
- **Text-to-Speech**: Automatic audio playback of recognized text
- **Voice Guidance for Blind Users**: Real-time audio instructions and feedback to help position braille text in the detection box
- **AI-Powered Instructions**: Context-aware guidance using Google Gemini API
- **Confidence Scoring**: Real-time confidence metrics for recognition accuracy

## 📁 Project Structure

```
NEW2/
├── backend/                 # FastAPI backend server
│   ├── main.py              # Main API server
│   ├── best_ctc_inference_2.keras  # Trained CRNN model (not in git)
│   ├── charset.txt          # Character vocabulary (A-Z)
│   ├── requirements.txt    # Python dependencies
│   ├── new_braille.ipynb   # Model training notebook
│   └── words.ipynb         # Inference/testing notebook
│
├── frontend/                # React + Vite frontend
│   ├── src/
│   │   ├── App.jsx         # Main application component
│   │   ├── components/
│   │   │   ├── CameraCapture.jsx    # Camera capture component
│   │   │   ├── ImageUpload.jsx      # Image upload component
│   │   │   ├── ResultDisplay.jsx    # Results display component
│   │   │   ├── TextToSpeech.jsx     # TTS component
│   │   │   └── VoiceGuidance.jsx    # Voice guidance for blind users
│   │   └── ...
│   ├── package.json        # Node.js dependencies
│   └── vite.config.js      # Vite configuration
│
└── README.md               # This file
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** with pip
- **Node.js 16+** and npm
- **Google Gemini API Key** (for word correction)

### Backend Setup

1. Navigate to the backend directory:

```bash
cd backend
```

2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

3. Ensure model files are present:

   - `best_ctc_inference_2.keras` - Trained model file (download separately)
   - `charset.txt` - Character vocabulary file (included)

4. Configure Gemini API key in `main.py`:

   - Set the `GEMINI_API_KEY` environment variable, **or**
   - Update the `GEMINI_API_KEY` variable with your key (default placeholder: `PLEASE ENTER YOUR API KEY HERE`)

5. Start the backend server:

```bash
python main.py
```

Or using uvicorn:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The backend will run on `http://localhost:8000`

### Frontend Setup

1. Navigate to the frontend directory:

```bash
cd frontend
```

2. Install Node.js dependencies:

```bash
npm install
```

3. Start the development server:

```bash
npm run dev
```

The frontend will run on `http://localhost:5173`

## 🔧 Configuration

### Backend Configuration

Edit `backend/main.py` to configure:

- **Model Path**: `MODEL_PATH` - Path to the Keras model file
- **Charset Path**: `CHARSET_PATH` - Path to character vocabulary
- **Gemini API Key**: `GEMINI_API_KEY` - Your Google Gemini API key
- **Gemini Model**: `GEMINI_MODEL` - Model version (default: `gemini-2.5-flash`)
- **Image Dimensions**: `IMG_HEIGHT` (64px), `MAX_IMG_WIDTH` (640px)
- **Beam Width**: `BEAM_WIDTH` (1 = greedy decoding)

### Frontend Configuration

Edit `frontend/vite.config.js` to configure:

- **Backend Proxy**: Update proxy target if backend runs on different port
- **Detection Box Size**: Edit `CameraCapture.jsx` to adjust detection box dimensions

## 📡 API Endpoints

### Health Check

- `GET /` - Root endpoint with API information
- `GET /health` - Health check endpoint
- `GET /test` - Test endpoint to verify backend connectivity

### Recognition

- `POST /api/recognize` - Recognize braille text from image

### Voice Guidance

- `GET /api/camera-instructions` - Get AI-generated instructions for positioning braille text
  - **Query Parameters**:
    - `context` (string): Context for instructions (`initial`, `low_confidence`, `no_text_detected`)
    - `voice_mode` (boolean): If true, returns shorter instructions optimized for voice output
  - **Response**:
    ```json
    {
      "instructions": "Position your braille text...",
      "context": "initial"
    }
    ```
  - **Body**: Form data with `photo` field (image file)
  - **Query Parameters**:
    - `is_camera` (boolean): Whether image is from camera (default: false)
    - `use_enhancement` (boolean): Image enhancement flag (default: false)
  - **Response**:
    ```json
    {
      "word": "RECOGNIZED_TEXT",
      "confidence": 0.95,
      "original": "ORIGINAL_TEXT" (if corrected),
      "corrected": true/false
    }
    ```

## 🔄 Processing Pipeline

### Camera Mode Flow:

```
Video Frame
    ↓
Frontend: Detection Box Crop (60% width × 50% height)
    ↓
Backend: Auto-crop (braille region detection)
    ↓
Backend: Preprocess (resize, normalize to [-1, 1])
    ↓
Backend: Model Inference (CRNN)
    ↓
Backend: CTC Decode (greedy decoding)
    ↓
Backend: Gemini Correction (word validation & correction)
    ↓
Frontend: Display Result + Text-to-Speech
```

### Upload Mode Flow:

```
Uploaded Image
    ↓
Backend: Auto-crop (braille region detection)
    ↓
Backend: Preprocess (resize, normalize to [-1, 1])
    ↓
Backend: Model Inference (CRNN)
    ↓
Backend: CTC Decode (greedy decoding)
    ↓
Backend: Gemini Correction (word validation & correction)
    ↓
Frontend: Display Result + Text-to-Speech
```

## 🧠 Model Details

- **Architecture**: CRNN (Convolutional Recurrent Neural Network)
- **Loss Function**: CTC (Connectionist Temporal Classification)
- **Input**: Grayscale images, 64px height, max 640px width
- **Output**: Character sequence (A-Z)
- **Decoding**: Greedy CTC decoding (beam_width=1)

## 🔊 Voice Guidance & Accessibility

The system includes comprehensive voice guidance features designed specifically for blind and visually impaired users:

### Voice Guidance Features

- **Initial Instructions**: When the camera starts, the system provides clear audio instructions on how to position braille text
- **Real-time Feedback**: Audio feedback based on recognition confidence:
  - **High Confidence (≥0.8)**: Success beep + "Text detected: [word]. Position is good."
  - **Medium Confidence (0.6-0.8)**: Info beep + detected text
  - **Low Confidence (<0.6)**: Warning beep + guidance to improve positioning
- **No Detection Guidance**: Helpful instructions when no text is detected after multiple attempts
- **Position Improvement Feedback**: Audio cues when positioning improves or worsens
- **Periodic Tips**: Helpful positioning tips every 15 seconds if having issues
- **Audio Cues**: Different beep patterns for success, warning, error, and info

### Voice Guidance Controls

- **Toggle Button**: Users can enable/disable voice guidance with a single button
- **Speech Queue**: Messages are queued to prevent overlapping speech
- **Voice Optimization**: Instructions are optimized for text-to-speech when in voice mode

## 🛠️ Technologies Used

### Backend

- **FastAPI** - Modern Python web framework
- **TensorFlow/Keras** - Deep learning framework
- **Pillow (PIL)** - Image processing
- **OpenCV** - Image preprocessing and cropping
- **Google Generative AI** - Word correction API and instruction generation
- **Uvicorn** - ASGI server

### Frontend

- **React 18** - UI framework
- **Vite** - Build tool and dev server
- **Web Speech API** - Text-to-speech functionality and voice guidance
- **Web Audio API** - Audio cues and beeps for feedback
- **MediaDevices API** - Camera access

## 📝 Development

### Running in Development Mode

**Backend** (with auto-reload):

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Frontend** (with hot reload):

```bash
cd frontend
npm run dev
```

### Building for Production

**Frontend**:

```bash
cd frontend
npm run build
```

The built files will be in `frontend/dist/`

## 🐛 Troubleshooting

### Camera Not Working

- Ensure camera permissions are granted in browser
- Check if camera is being used by another application
- Try refreshing the page or restarting the browser

### Backend Connection Errors

- Verify backend is running on `http://localhost:8000`
- Check CORS settings in `backend/main.py`
- Ensure frontend proxy is configured correctly in `vite.config.js`

### Model Not Loading

- Verify `best_ctc_inference_2.keras` exists in `backend/` directory
- Check file permissions
- Ensure TensorFlow is installed correctly

### Gemini API Errors

- Verify API key is correct and has sufficient quota
- Check internet connection
- Review backend logs for specific error messages

## 🙏 Acknowledgments & Credits

- TensorFlow/Keras for deep learning framework
- Google Gemini for word correction API
- FastAPI for backend framework
- React and Vite for frontend development
- Braille word recognition model architecture inspired by and adapted from  
  [`braille-words-detection` by AbhijithBabu12](https://github.com/AbhijithBabu12/braille-words-detection)

---

**Note**: The model file (`best_ctc_inference_2.keras`) is not included in the repository due to size constraints. Please ensure you have the trained model file in the `backend/` directory before running the application.

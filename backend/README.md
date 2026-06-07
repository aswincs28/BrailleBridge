# Braille Recognition Backend API

FastAPI server for Braille text recognition using a trained CRNN model.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure model and charset files are present:
   - `best_ctc_inference_2.keras` - Trained model file
   - `charset.txt` - Character vocabulary file

## Running the Server

### Option 1: Using Python directly
```bash
python main.py
```

### Option 2: Using uvicorn
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The server will start on `http://localhost:8000`

## API Endpoints

### Health Check
- `GET /` - Root endpoint with API info
- `GET /health` - Health check endpoint

### Recognition
- `POST /api/recognize` - Recognize braille text from image
  - **Body**: Form data with `photo` field (image file)
  - **Query params**: `use_enhancement` (boolean, default: false)
  - **Response**: 
    ```json
    {
      "word": "RECOGNIZED_TEXT",
      "confidence": 0.95
    }
    ```

## Example Usage

```bash
curl -X POST "http://localhost:8000/api/recognize?use_enhancement=false" \
  -F "photo=@path/to/braille_image.jpg"
```

## Notes

- The model expects images to be preprocessed: 64px height, max 640px width, normalized to [-1, 1]
- Uses CTC decoding with beam search (beam_width=10) for text recognition
- Confidence score is computed based on average probability of predicted characters


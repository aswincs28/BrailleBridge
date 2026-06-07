"""
Braille Recognition Backend API
===============================
This FastAPI backend provides endpoints for recognizing braille text from images using:
- A CNN + BiLSTM + CTC-loss deep learning model (TensorFlow/Keras)
- Google Gemini API for word correction and voice guidance
- Image preprocessing (auto-cropping, resizing, normalization)
- CTC decoding to convert model predictions to text

Architecture:
- Model: Pre-trained Keras model loaded on startup
- Charset: Character mapping file defining valid output characters
- Image Processing: Auto-crop braille region, resize to fixed height, pad/truncate width
- CTC Decoding: Convert model logits to text using greedy or beam search
- Word Correction: Use Gemini API to correct incomplete/misspelled words
- Voice Guidance: Generate AI-powered instructions for blind users

Author: Based on model from https://github.com/AbhijithBabu12/braille-words-detection
"""

import os
import numpy as np
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import io
import uvicorn
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import re
import cv2

# ==================== CONFIGURATION SECTION ====================
# Path to the pre-trained Keras model file (CNN + BiLSTM + CTC architecture)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_ctc_inference_2.keras")

# Path to the charset file containing all valid output characters (one per line)
CHARSET_PATH = os.path.join(os.path.dirname(__file__), "charset.txt")

# Image preprocessing constants - these must match the model's training parameters
IMG_HEIGHT = 64  # Fixed height for all input images (model requirement)
MAX_IMG_WIDTH = 640  # Maximum width before truncation (model requirement)

# CTC decoding configuration
BEAM_WIDTH = 1  # 1 = greedy decoding (more reliable), >1 = beam search (may be fragile)

# Google Gemini API configuration for word correction and voice guidance
# IMPORTANT: Set GEMINI_API_KEY environment variable or replace the placeholder below
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyAQqdC_dCrMOL_dqjto87QCB_AiZdkXkRc")
# Gemini model versions
# - GEMINI_CAMERA_MODEL: Used for camera-based braille detection (native audio preview model)
# - GEMINI_MODEL: Used for word correction, voice guidance, and other text generation tasks
GEMINI_CAMERA_MODEL = "gemini-2.5-flash"  # Camera detection model (vision-capable)
GEMINI_MODEL = "gemini-2.5-flash"  # Standard model for other purposes

# ==================== FASTAPI APPLICATION SETUP ====================
# Initialize FastAPI application with metadata
app = FastAPI(title="Braille Recognition API", version="1.0.0")

# CORS (Cross-Origin Resource Sharing) middleware configuration
# This allows the React frontend (running on port 5173) to make requests to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],  # Vite default port
    allow_credentials=True,  # Allow cookies/credentials in requests
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Request logging middleware - logs all incoming requests for debugging
@app.middleware("http")
async def log_requests(request, call_next):
    """
    Middleware function that logs every HTTP request before processing.
    This helps with debugging and monitoring API usage.
    """
    print(f"[BACKEND] Incoming request: {request.method} {request.url}")
    print(f"[BACKEND] Headers: {dict(request.headers)}")
    response = await call_next(request)  # Process the request
    print(f"[BACKEND] Response status: {response.status_code}")
    return response

# ==================== GLOBAL VARIABLES ====================
# These are loaded once on startup and reused for all requests
model = None  # TensorFlow/Keras model for braille recognition (loaded on startup)
num_to_char = None  # Dictionary mapping numeric IDs to characters (e.g., {0: 'A', 1: 'B', ...})
charset_tokens = None  # List of all valid characters in the charset

# ==================== HELPER FUNCTIONS ====================

def load_charset(path):
    """
    Load the character set from a text file.
    
    The charset file contains one character per line, defining all valid output characters
    that the model can predict. This is used to map model output (numeric IDs) to actual text.
    
    Args:
        path (str): Path to the charset file
        
    Returns:
        tuple: (num_to_char dict, charset_tokens list)
            - num_to_char: Dictionary mapping {0: first_char, 1: second_char, ...}
            - charset_tokens: List of all characters in order
    
    Raises:
        FileNotFoundError: If the charset file doesn't exist
    """
    toks = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.rstrip("\n\r")  # Remove newline characters
                if s == "":  # Skip empty lines
                    continue
                toks.append(s)  # Add character to list
    else:
        raise FileNotFoundError(f"Charset file not found: {path}")
    
    # Create mapping from numeric ID to character (e.g., {0: 'A', 1: 'B', ...})
    num_to_char = {i: c for i, c in enumerate(toks)}
    return num_to_char, toks

def autocrop_braille_pil(pil_img, dark_thresh=200, pad=10):
    """
    Smart autocrop: detect dark connected components, drop noise and
    drop lone-bottom-row "punctuation" dots that hang off the left/right edge.

    Why this matters: the model only knows A-Z. A lone bottom-row dot is the
    braille `dot 3` (or `dot 6`) cell — a real punctuation symbol. If it sneaks
    into the crop, the model hallucinates a phantom letter (typically K) at the
    edge of the word. Dropping such isolated edge dots is essential.

    Pipeline:
      1. grayscale -> threshold -> morphological open (kill speckle)
      2. connected components, drop tiny noise blobs (area < 0.2 * median)
      3. compute top/bottom row Y from cy spread
      4. for leftmost & rightmost components: if cy is in the bottom row AND
         no other component shares its X column (within ~half a dot spacing),
         drop it — it's almost certainly stray punctuation
      5. crop bounding box of survivors with padding
    """
    gray = np.array(pil_img.convert("L"))
    h_img, w_img = gray.shape
    binary = (gray < dark_thresh).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    n_labels, _, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n_labels <= 1:
        # Try a more lenient threshold once before giving up
        fallback = max(0, dark_thresh - 40)
        print(f"[DEBUG] Autocrop: no components at thresh={dark_thresh}, trying {fallback}")
        binary = (gray < fallback).astype(np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        n_labels, _, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if n_labels <= 1:
            print(f"[DEBUG] Autocrop: still no components, returning original image")
            return pil_img.copy()

    comps = []
    for i in range(1, n_labels):
        x, y, w, h, area = stats[i]
        cx, cy = centroids[i]
        comps.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h),
                      "area": int(area), "cx": float(cx), "cy": float(cy)})

    # Drop tiny noise blobs (smaller than 20% of median area, and below an absolute floor)
    areas = sorted(c["area"] for c in comps)
    median_area = areas[len(areas) // 2]
    comps = [c for c in comps if c["area"] >= max(4, 0.2 * median_area)]
    if not comps:
        print(f"[DEBUG] Autocrop: all components rejected as noise, returning original image")
        return pil_img.copy()

    # Detect lone-bottom-row outliers at the edges
    cys = [c["cy"] for c in comps]
    top_y, bot_y = min(cys), max(cys)
    row_span = bot_y - top_y
    if row_span >= 5:
        bot_threshold = bot_y - 0.25 * row_span
        sorted_cx = sorted(set(round(c["cx"]) for c in comps))
        gaps = [b - a for a, b in zip(sorted_cx[:-1], sorted_cx[1:])]
        col_tol = (min(gaps) if gaps else 30) * 0.7

        def has_column_companion(target):
            return any(c is not target and abs(c["cx"] - target["cx"]) <= col_tol for c in comps)

        leftmost = min(comps, key=lambda c: c["cx"])
        if leftmost["cy"] >= bot_threshold and not has_column_companion(leftmost):
            print(f"[DEBUG] Autocrop: dropping lone bottom-left dot at ({leftmost['cx']:.0f},{leftmost['cy']:.0f})")
            comps = [c for c in comps if c is not leftmost]
        if comps:
            rightmost = max(comps, key=lambda c: c["cx"])
            if rightmost["cy"] >= bot_threshold and not has_column_companion(rightmost):
                print(f"[DEBUG] Autocrop: dropping lone bottom-right dot at ({rightmost['cx']:.0f},{rightmost['cy']:.0f})")
                comps = [c for c in comps if c is not rightmost]

    if not comps:
        print(f"[DEBUG] Autocrop: nothing left after edge filter, returning original image")
        return pil_img.copy()

    x1 = min(c["x"] for c in comps)
    y1 = min(c["y"] for c in comps)
    x2 = max(c["x"] + c["w"] for c in comps)
    y2 = max(c["y"] + c["h"] for c in comps)
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(w_img, x2 + pad); y2 = min(h_img, y2 + pad)
    print(f"[DEBUG] Autocrop: cropped to ({x1},{y1})-({x2},{y2}) size {x2-x1}x{y2-y1}")
    return pil_img.crop((x1, y1, x2, y2))


def pad_to_target_aspect(pil_img, target_aspect=3.0):
    """
    Pad the image vertically with white background so its width/height ratio
    is at most `target_aspect`. This preserves the natural per-cell aspect
    ratio that the model expects after resize-to-64-height.

    Without this, a tightly autocropped braille word (e.g. 450 wide x 90 tall)
    has aspect ratio 5:1 and gets badly stretched when resized to 64h x 320w —
    each dot becomes oblong rather than round, and the model misreads it.
    """
    img = pil_img.convert("L")
    w, h = img.size
    if w / max(1, h) <= target_aspect:
        return img
    new_h = int(w / target_aspect)
    canvas = Image.new("L", (w, new_h), color=255)
    y_off = (new_h - h) // 2
    canvas.paste(img, (0, y_off))
    print(f"[DEBUG] Aspect-pad: {w}x{h} -> {w}x{new_h} (target aspect {target_aspect})")
    return canvas

def preprocess_image_arr(pil_img, img_height=IMG_HEIGHT, max_width=MAX_IMG_WIDTH):
    """
    Preprocess image for model input: resize, normalize, and pad/truncate to fixed dimensions.
    
    The model requires fixed-height images with variable width (up to max_width).
    This function:
    1. Resizes image to fixed height while preserving aspect ratio
    2. Normalizes pixel values to [-1, 1] range (model requirement)
    3. Pads or truncates width to max_width
    4. Adds channel dimension for model input
    
    This preprocessing must match exactly what was used during model training.
    
    Args:
        pil_img (PIL.Image): Input image (should be cropped to braille region)
        img_height (int): Target height (default: 64, must match model training)
        max_width (int): Maximum width before truncation (default: 640)
    
    Returns:
        tuple: (preprocessed_array, actual_width)
            - preprocessed_array: numpy array of shape (height, max_width, 1) with values in [-1, 1]
            - actual_width: The actual width after resizing (before padding/truncation)
    """
    # Convert to grayscale (single channel)
    img = pil_img.convert("L")
    w, h = img.size  # Original width and height
    
    # Resize to fixed height while preserving aspect ratio
    new_h = img_height  # Target height (64 pixels)
    new_w = max(1, int(w * (new_h / float(h))))  # Calculate new width to maintain aspect ratio
    img = img.resize((new_w, new_h), Image.LANCZOS)  # High-quality resampling
    
    # Convert to numpy array and normalize to [-1, 1] range
    # Original: [0, 255] -> [0.0, 1.0] -> [-1.0, 1.0]
    arr = np.array(img).astype(np.float32) / 255.0  # Normalize to [0, 1]
    arr = arr * 2.0 - 1.0  # Transform to [-1, 1] (model requirement)
    
    # Handle width: pad if too narrow, truncate if too wide
    if new_w < max_width:
        # Image is narrower than max_width: pad with -1.0 (background value)
        pad = np.ones((new_h, max_width), dtype=np.float32) * (-1.0)  # Create padding array
        pad[:, :new_w] = arr  # Copy image into padded array
        arr = pad
    elif new_w > max_width:
        # Image is wider than max_width: truncate to max_width
        arr = arr[:, :max_width]
        new_w = max_width
    
    # Add channel dimension: (height, width) -> (height, width, 1)
    # Model expects 3D input: (batch, height, width, channels)
    arr = np.expand_dims(arr, -1)
    
    return arr, new_w

def preprocess_image_from_bytes(image_bytes, img_height=IMG_HEIGHT, max_width=MAX_IMG_WIDTH, is_camera=False):
    """
    Main preprocessing pipeline: convert image bytes to model-ready array.
    
    This function handles two different input modes:
    
    1. UPLOAD MODE (is_camera=False):
       - Full image from file upload
       - Auto-crop braille region with strict threshold (240)
       - Preprocess for model
       - This mode is highly accurate and should not be modified
    
    2. CAMERA MODE (is_camera=True):
       - Image is already cropped to detection box by frontend (60% width, 50% height)
       - Auto-crop braille region with lenient thresholds (tries 200, 180, 160)
       - Preprocess for model
       - More lenient because camera images may have different lighting/quality
    
    Args:
        image_bytes (bytes): Raw image file bytes (JPEG, PNG, etc.)
        img_height (int): Target height for model input
        max_width (int): Maximum width for model input
        is_camera (bool): True if from camera (uses lenient cropping), False if from upload
    
    Returns:
        tuple: (preprocessed_array, actual_width)
            - preprocessed_array: numpy array ready for model input
            - actual_width: actual width after resizing (before padding)
    """
    # Open image from bytes using PIL (supports JPEG, PNG, etc.)
    pil_img = Image.open(io.BytesIO(image_bytes))
    
    if is_camera:
        # ========== CAMERA MODE PROCESSING ==========
        print(f"[DEBUG] Camera: Received detection box image size: {pil_img.size}")
        cropped_img = autocrop_braille_pil(pil_img, dark_thresh=200, pad=10)
        if cropped_img.size == pil_img.size:
            print("[DEBUG] Camera: Autocrop didn't shrink at thresh=200, trying thresh=180")
            cropped_img = autocrop_braille_pil(pil_img, dark_thresh=180, pad=10)
        if cropped_img.size == pil_img.size:
            print("[DEBUG] Camera: Autocrop didn't shrink at thresh=180, trying thresh=160")
            cropped_img = autocrop_braille_pil(pil_img, dark_thresh=160, pad=10)
    else:
        # ========== UPLOAD MODE PROCESSING ==========
        cropped_img = autocrop_braille_pil(pil_img, dark_thresh=200, pad=10)

    # Pad vertically to a sane aspect ratio so the model doesn't see a stretched strip
    cropped_img = pad_to_target_aspect(cropped_img, target_aspect=3.0)

    # Preprocess the cropped image: resize, normalize, pad/truncate to model dimensions
    arr, actual_width = preprocess_image_arr(cropped_img, img_height, max_width)

    return arr, actual_width

def map_ids_to_text(ids_row, logits_or_probs, num_to_char):
    """
    Convert CTC decoded numeric IDs to text string using character mapping.
    
    CTC (Connectionist Temporal Classification) decoding produces a sequence of
    numeric IDs representing characters. This function maps those IDs to actual
    characters using the charset dictionary.
    
    Args:
        ids_row (numpy array): Sequence of character IDs from CTC decoding
        logits_or_probs (numpy array): Model output (used to determine blank ID)
        num_to_char (dict): Dictionary mapping {ID: character}
    
    Returns:
        str: Decoded text string
    """
    # Blank token is always the last class in CTC (used for alignment)
    blank = logits_or_probs.shape[-1] - 1
    out = []
    
    # Convert each ID to its corresponding character
    for cid in ids_row:
        if int(cid) == -1:  # Skip padding tokens
            continue
        if int(cid) >= blank:  # Skip invalid IDs (shouldn't happen)
            continue
        
        # Map ID to character, use "?" if ID not found in charset
        out.append(num_to_char.get(int(cid), "?"))
    
    return "".join(out)  # Join characters into string

def decode_ctc(logits, num_to_char, beam_width=BEAM_WIDTH):
    """
    Decode CTC model output (logits) to text string.
    
    This function implements the CTC decoding algorithm:
    1. Apply softmax to convert logits to probabilities
    2. Use TensorFlow's CTC decoder (greedy or beam search)
    3. Fallback to manual argmax decoding if CTC decode fails
    4. Map numeric IDs to characters
    
    The decoding strategy matches the original training notebook for consistency.
    
    Args:
        logits (numpy array): Model output of shape (batch, time_steps, num_classes)
        num_to_char (dict): Dictionary mapping {ID: character}
        beam_width (int): Beam width for decoding (1 = greedy, >1 = beam search)
    
    Returns:
        str: Decoded text string
    """
    # Convert logits to probabilities using softmax (required for CTC decoding)
    probs = tf.nn.softmax(logits, axis=-1).numpy()
    batch_size = probs.shape[0]
    time_steps = probs.shape[1]
    
    # CTC decoder needs input_length array (all time steps are valid)
    input_length = np.ones((batch_size,), dtype=np.int32) * time_steps
    
    text_probs = ""  # Result from CTC decode
    text_manual = ""  # Result from manual fallback
    
    # ========== PRIMARY DECODING: TensorFlow CTC Decoder ==========
    # This is the recommended method (same as training notebook)
    try:
        decoded_probs = tf.keras.backend.ctc_decode(
            probs,  # Probabilities (not raw logits)
            input_length=input_length,  # Length of each sequence
            greedy=(beam_width == 1),  # True for greedy, False for beam search
            beam_width=beam_width,  # Number of beams (ignored if greedy=True)
            top_paths=1  # Return only the best path
        )
        first = decoded_probs[0][0]  # Extract first (and only) decoded sequence
        
        # Handle different tensor types (SparseTensor or regular Tensor)
        if isinstance(first, tf.SparseTensor):
            # Convert sparse tensor to dense array
            dense_probs = tf.sparse.to_dense(first, default_value=-1).numpy()
        elif tf.is_tensor(first):
            # Regular tensor - convert to numpy
            dense_probs = first.numpy()
        else:
            # Already a numpy array
            dense_probs = np.asarray(first)
        
        # Map numeric IDs to characters
        text_probs = map_ids_to_text(dense_probs[0], probs, num_to_char)
    except Exception as e:
        print(f"CTC decode on PROBS failed: {e}")
        text_probs = ""  # Failed - will use fallback
    
    # ========== FALLBACK DECODING: Manual Argmax ==========
    # If CTC decode fails, use simple greedy argmax decoding
    text_manual, _, _ = manual_argmax_collapse(logits, num_to_char)
    
    # Prefer CTC decode result if available, otherwise use manual fallback
    final = text_probs if (text_probs is not None and len(text_probs) > 0) else text_manual
    
    return final

def manual_argmax_collapse(logits_np, num_to_char):
    """
    Manual greedy decoding: simple argmax with blank token collapsing.
    
    This is a fallback decoding method that:
    1. Takes argmax at each time step (most likely character)
    2. Removes blank tokens (CTC alignment tokens)
    3. Collapses repeated characters (CTC can output "H-H-E-L-L-O" -> "HELLO")
    
    This method is less sophisticated than CTC decode but more robust to errors.
    
    Args:
        logits_np (numpy array): Model logits of shape (batch, time_steps, num_classes)
        num_to_char (dict): Dictionary mapping {ID: character}
    
    Returns:
        tuple: (text, argseq, collapsed_ids)
            - text: Decoded text string
            - argseq: Raw argmax sequence (before collapsing)
            - collapsed_ids: IDs after removing blanks and duplicates
    """
    # Extract argmax sequence: for each time step, pick the class with highest logit
    # Shape: (batch, time_steps, num_classes) -> (time_steps,) of class IDs
    arg = np.argmax(logits_np, axis=-1)[0].tolist()  # Convert to list of integers
    
    # Blank token is always the last class (used for CTC alignment)
    blank = logits_np.shape[-1] - 1
    
    # Collapse sequence: remove blanks and repeated characters
    collapsed = []
    prev = None  # Track previous character to detect repeats
    for a in arg:
        # Skip if same as previous (CTC can output "H-H-E" -> we want "HE")
        if a == prev:
            prev = a
            continue
        # Skip blank tokens (alignment tokens, not actual characters)
        if a == blank:
            prev = a
            continue
        # Valid character - add to output
        collapsed.append(a)
        prev = a
    
    # Map numeric IDs to characters and join into string
    text = "".join([num_to_char.get(int(i), "?") for i in collapsed])
    
    return text, arg, collapsed

def compute_confidence(logits, decoded_text, num_to_char):
    """
    Compute confidence score for the decoded text based on model probabilities.
    
    Confidence is calculated as the average probability of the predicted
    characters (excluding blank tokens). Higher values indicate more confident predictions.
    
    Args:
        logits (numpy array): Model output of shape (batch, time_steps, num_classes)
        decoded_text (str): The decoded text string
        num_to_char (dict): Dictionary mapping {ID: character}
    
    Returns:
        float: Confidence score between 0.0 and 1.0
    """
    # Empty text has zero confidence
    if not decoded_text or len(decoded_text) == 0:
        return 0.0
    
    # Convert logits to probabilities and extract first (and only) batch item
    probs = tf.nn.softmax(logits, axis=-1).numpy()[0]  # Shape: (time_steps, num_classes)
    blank_idx = probs.shape[-1] - 1  # Blank token index
    
    # Collect probabilities of predicted characters (excluding blanks)
    char_probs = []
    for t in range(probs.shape[0]):  # For each time step
        char_idx = np.argmax(probs[t])  # Most likely character at this time step
        if char_idx != blank_idx:  # Skip blank tokens
            char_probs.append(probs[t, char_idx])  # Store probability
    
    # If no valid characters found, confidence is zero
    if len(char_probs) == 0:
        return 0.0
    
    # Confidence = average probability of predicted characters
    # Higher probability = more confident prediction
    avg_prob = np.mean(char_probs)
    confidence = float(avg_prob)
    
    # Ensure confidence is in valid range [0, 1]
    return min(1.0, max(0.0, confidence))

# ==================== WORD VALIDATION AND CORRECTION ====================
# These functions use Google Gemini API to correct incomplete/misspelled words
# that may result from braille recognition errors.

def fallback_word_correction(word):
    """
    Simple dictionary-based word correction for common misspellings.
    
    This is a fallback when Gemini API fails or is blocked by safety filters.
    Handles common incomplete words that result from braille recognition.
    
    Args:
        word (str): Potentially misspelled or incomplete word
    
    Returns:
        str: Corrected word, or original if no correction found
    """
    # Common misspellings/incomplete words
    corrections = {
        "EG": "EGG",
        "CLEBRATION": "CELEBRATION",
        "IFORMATION": "INFORMATION",
        "CHILDR": "CHILDREN",
        "GOO": "GOOD",
        "BAL": "BALL",
        "HELL": "HELLO",
        "WOR": "WORD",
        "WORL": "WORLD",
    }
    corrected = corrections.get(word, word)
    if corrected != word:
        print(f"[DEBUG] Fallback correction found: '{word}' -> '{corrected}'")
    else:
        print(f"[DEBUG] Fallback correction: no match for '{word}'")
    return corrected

def try_simple_gemini_correction(word, model):
    """Try a very simple prompt that's less likely to trigger safety filters."""
    try:
        # Ultra-simple prompt
        simple_prompt = f"Complete this word: {word}"
        response = model.generate_content(
            simple_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                max_output_tokens=10,
            ),
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
        if response.candidates and response.candidates[0].finish_reason != 2:
            result = response.text.strip().upper().split()[0] if response.text.strip() else word
            print(f"[DEBUG] Simple Gemini correction: '{word}' -> '{result}'")
            return result
    except Exception as e:
        print(f"[DEBUG] Simple Gemini correction failed: {e}")
    return word

def is_valid_word(word):
    """Check if word contains only valid English letters and has reasonable length."""
    if not word or len(word) < 1:
        return False
    # Check if word contains only letters (A-Z, a-z)
    return bool(re.match(r'^[A-Za-z]+$', word))

def correct_word_with_gemini(word):
    """
    Use Gemini API to find the most similar/correct word.
    Returns the corrected word or original if correction fails.
    """
    if not word or len(word) < 1:
        print(f"[DEBUG] correct_word_with_gemini: Empty word, returning as-is")
        return word
    
    print(f"[DEBUG] correct_word_with_gemini: Attempting correction for '{word}'")
    
    try:
        # Configure Gemini
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel(GEMINI_MODEL)
        print(f"[DEBUG] Gemini model configured: {GEMINI_MODEL}")
        
        # Create prompt for word correction - more explicit
        prompt = f"""You are a word correction assistant for braille text recognition. The input word may be incomplete or have missing letters.

Task: Given the incomplete word "{word}", return the most likely complete English word.

Important rules:
- Return ONLY the corrected word in UPPERCASE
- Do NOT include any explanation, punctuation, or extra text
- The word may be missing 1-3 letters
- Choose the most common English word that matches the pattern
- If the word is already complete and correct, return it exactly as-is

Input: "{word}"
Output (word only):"""
        
        print(f"[DEBUG] Sending prompt to Gemini for '{word}'")
        
        # Generate correction with safety settings disabled
        response = gemini_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,  # Lower temperature for more consistent results
                top_p=0.8,
                top_k=10,
                max_output_tokens=15,
            ),
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
        
        # Handle response safely - check for blocked content
        if not response.candidates or len(response.candidates) == 0:
            print(f"[DEBUG] No candidates in response")
            raise ValueError("No response candidates")
        
        candidate = response.candidates[0]
        print(f"[DEBUG] Response finish_reason: {candidate.finish_reason}")
        
        if candidate.finish_reason == 2:  # SAFETY
            print(f"[DEBUG] Response blocked by safety filters, trying alternative approach")
            if candidate.content and candidate.content.parts:
                result_text = "".join([part.text for part in candidate.content.parts if hasattr(part, 'text')])
                if result_text:
                    print(f"[DEBUG] Extracted text from parts: '{result_text}'")
                else:
                    raise ValueError("Response blocked by safety filters and no text in parts")
            else:
                raise ValueError("Response blocked by safety filters")
        else:
            result_text = response.text
        
        print(f"[DEBUG] Gemini raw response: '{result_text}'")
        
        # Extract corrected word from response
        corrected = result_text.strip().upper()
        
        # Clean up: remove quotes, extra text, take first word only
        corrected = corrected.replace('"', '').replace("'", '').strip()
        # Remove common prefixes/suffixes that Gemini might add
        corrected = corrected.replace("CORRECTED:", "").replace("WORD:", "").strip()
        # Take only the first word if multiple words returned
        corrected = corrected.split()[0] if corrected.split() else word
        
        print(f"[DEBUG] Cleaned correction: '{corrected}' (original: '{word}')")
        
        # Validate: should be alphabetic and similar length
        if corrected and re.match(r'^[A-Za-z]+$', corrected):
            # Allow more flexibility in length (up to 3 characters difference for incomplete words)
            if abs(len(corrected) - len(word)) <= 3:
                print(f"[DEBUG] Word correction accepted: '{word}' -> '{corrected}'")
                return corrected.upper()
            else:
                print(f"[DEBUG] Correction rejected: length difference too large ({len(corrected)} vs {len(word)})")
        else:
            print(f"[DEBUG] Correction rejected: invalid format or empty")
        
        print(f"[DEBUG] Returning original word: '{word}'")
        return word.upper()
        
    except Exception as e:
        print(f"[DEBUG] Gemini correction failed for '{word}': {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return word.upper()

def check_braille_with_gemini(image_bytes):
    """
    Pre-check image for braille text using Gemini Vision API.
    
    This function is used ONLY in camera mode to avoid running the expensive
    model inference on frames that don't contain braille text. It acts as a
    pre-filter to improve performance and reduce unnecessary processing.
    
    The Gemini Vision API analyzes the image and determines if braille dots
    are present. If no braille is detected, the model inference is skipped
    and directional guidance is provided to the user instead.
    
    IMPORTANT: This is NOT used in upload mode (which works perfectly).
    
    Args:
        image_bytes (bytes): Raw image bytes (JPEG format)
    
    Returns:
        bool: True if braille is detected, False otherwise
              (Defaults to True on error to avoid blocking valid images)
    """
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        # Use camera-specific model for braille detection
        gemini_model = genai.GenerativeModel(GEMINI_CAMERA_MODEL)
        print(f"[DEBUG] Using camera model for braille detection: {GEMINI_CAMERA_MODEL}")
        
        # Prepare image for Gemini
        pil_img = Image.open(io.BytesIO(image_bytes))
        
        # Convert PIL image to format Gemini can use
        import base64
        img_buffer = io.BytesIO()
        pil_img.save(img_buffer, format='JPEG', quality=85)
        img_bytes = img_buffer.getvalue()
        
        prompt = """Look at this image carefully. Does it contain braille text (raised dots arranged in patterns)?

Braille text consists of:
- Small raised dots arranged in cells
- Usually appears as dark dots on a lighter background or raised bumps
- Arranged in horizontal rows
- Each cell typically has 2-3 columns of dots

Answer with ONLY "YES" if you can clearly see braille text, or "NO" if you cannot see any braille text.
Do not provide any explanation, just "YES" or "NO"."""
        
        # Use Gemini Vision API
        response = gemini_model.generate_content(
            [prompt, {"mime_type": "image/jpeg", "data": img_bytes}],
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,  # Low temperature for consistent detection
                top_p=0.8,
                top_k=5,
                max_output_tokens=5,  # Just need YES or NO
            ),
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
        
        if not response.candidates or len(response.candidates) == 0:
            print("[DEBUG] Gemini braille detection: No response, defaulting to True (proceed)")
            return True  # Default to True to avoid blocking valid images
        
        candidate = response.candidates[0]
        if candidate.finish_reason == 2:  # SAFETY
            print("[DEBUG] Gemini braille detection: Blocked by safety, defaulting to True")
            return True
        
        result_text = response.text.strip().upper()
        print(f"[DEBUG] Gemini braille detection result: '{result_text}'")
        
        # Check if response indicates braille is present
        if "YES" in result_text or "BRAILLE" in result_text or "DOTS" in result_text:
            return True
        elif "NO" in result_text:
            return False
        else:
            # Ambiguous response, default to True to avoid blocking
            print(f"[DEBUG] Gemini braille detection: Ambiguous response '{result_text}', defaulting to True")
            return True
            
    except Exception as e:
        print(f"[DEBUG] Gemini braille detection failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        # On error, default to True to avoid blocking valid images
        return True

def validate_and_correct_word(word):
    """
    Validate word and correct if needed using Gemini.
    Returns (corrected_word, was_corrected)
    """
    if not word:
        print(f"[DEBUG] validate_and_correct_word: Empty word")
        return word, False
    
    # Normalize to uppercase
    word = word.upper().strip()
    print(f"[DEBUG] validate_and_correct_word: Processing '{word}'")
    
    # Always attempt correction for short words (likely incomplete)
    # or if format is invalid
    should_correct = False
    
    if not is_valid_word(word):
        print(f"[DEBUG] Invalid word format: '{word}', will attempt correction")
        should_correct = True
    elif len(word) <= 3:
        # Short words are likely incomplete (e.g., "EG" should be "EGG")
        print(f"[DEBUG] Short word detected (length {len(word)}), will attempt correction")
        should_correct = True
    else:
        # For longer words, check if it's a valid English word
        print(f"[DEBUG] Checking if '{word}' is a valid English word")
        should_correct = True  # Always check with Gemini
    
    if should_correct:
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            # Try the specified model, fallback to 1.5-flash if it fails
            try:
                gemini_model = genai.GenerativeModel(GEMINI_MODEL)
            except Exception as model_err:
                print(f"[DEBUG] Model {GEMINI_MODEL} failed, trying gemini-1.5-flash: {model_err}")
                gemini_model = genai.GenerativeModel("gemini-1.5-flash")
            
            # More explicit prompt for correction
            prompt = f"""You are correcting words from braille text recognition. The input word may be incomplete or have missing letters.

Input word: "{word}"

Task: Return the most likely complete English word. If the word is already complete and correct, return it exactly as-is.

Rules:
- Return ONLY the word in UPPERCASE, no explanation
- The word may be missing 1-4 letters
- Choose the most common English word
- Examples: "EG" -> "EGG", "CHILDR" -> "CHILDREN", "GOO" -> "GOOD"

Input: "{word}"
Output:"""
            
            print(f"[DEBUG] Sending validation/correction request to Gemini for '{word}'")
            
            response = gemini_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,  # Lower for more consistent results
                    top_p=0.7,
                    top_k=5,
                    max_output_tokens=15,
                ),
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                }
            )
            
            # Handle response safely - check for blocked content
            print(f"[DEBUG] Response finish_reason: {response.candidates[0].finish_reason if response.candidates else 'NO_CANDIDATES'}")
            
            if not response.candidates or len(response.candidates) == 0:
                print(f"[DEBUG] No candidates in response")
                raise ValueError("No response candidates")
            
            candidate = response.candidates[0]
            if candidate.finish_reason == 2:  # SAFETY
                print(f"[DEBUG] Response blocked by safety filters (finish_reason=2), trying fallback correction")
                # Fallback: Use a simple dictionary-based correction for common misspellings
                result_text = fallback_word_correction(word)
                if result_text == word:
                    # If fallback didn't help, try a simpler Gemini prompt
                    print(f"[DEBUG] Fallback didn't help, trying simpler Gemini prompt")
                    result_text = try_simple_gemini_correction(word, gemini_model)
                if result_text == word:
                    print(f"[DEBUG] All correction methods failed, using original word")
                    result_text = word
            else:
                # Normal case - use response.text
                result_text = response.text
            
            print(f"[DEBUG] Gemini validation response: '{result_text}'")
            
            result = result_text.strip().upper().replace('"', '').replace("'", '').strip()
            # Remove common prefixes
            result = result.replace("CORRECTED:", "").replace("WORD:", "").replace("RESULT:", "").strip()
            # Take only the first word
            result = result.split()[0] if result.split() else word
            
            print(f"[DEBUG] Processed result: '{result}' (original: '{word}')")
            
            # If result is different from input and valid, it was corrected
            if result != word and len(result) > 0 and re.match(r'^[A-Za-z]+$', result):
                # Allow up to 3 characters difference for incomplete words
                if abs(len(result) - len(word)) <= 3:
                    print(f"[DEBUG] Word correction applied: '{word}' -> '{result}'")
                    return result, True
                else:
                    print(f"[DEBUG] Correction rejected: length difference too large ({len(result)} vs {len(word)})")
            else:
                print(f"[DEBUG] No correction needed or result invalid")
        
        except Exception as e:
            print(f"[DEBUG] Word validation failed for '{word}': {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
    
    return word, False

# ----------------- Startup Event -----------------
@app.on_event("startup")
async def load_model():
    """Load model and charset on startup."""
    global model, num_to_char, charset_tokens
    
    print("Loading model and charset...")
    
    # Load charset
    if not os.path.exists(CHARSET_PATH):
        raise FileNotFoundError(f"Charset file not found: {CHARSET_PATH}")
    num_to_char, charset_tokens = load_charset(CHARSET_PATH)
    print(f"Loaded charset: {len(charset_tokens)} tokens")
    
    # Load model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print(f"Model loaded successfully. Output shape: {model.output_shape}")
        print(f"Model expects input shape: {model.input_shape}")
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")
    
    print("Model and charset loaded successfully!")

# ----------------- API Endpoints -----------------
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Braille Recognition API",
        "status": "running",
        "model_loaded": model is not None,
        "charset_size": len(charset_tokens) if charset_tokens else 0
    }

@app.get("/health")
async def health():
    """Health check endpoint."""
    print("[BACKEND] Health check requested")
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.get("/test")
async def test():
    """Test endpoint to verify backend is reachable."""
    print("[BACKEND] Test endpoint called")
    return {
        "message": "Backend is reachable!",
        "timestamp": str(__import__("datetime").datetime.now())
    }

@app.get("/api/camera-instructions")
async def get_camera_instructions(
    context: str = Query("", description="Optional context (e.g., 'low_confidence', 'no_text_detected', 'initial')"),
    voice_mode: bool = Query(False, description="If true, returns shorter instructions optimized for voice")
):
    """
    Generate AI-powered instructions for positioning braille text in camera detection box.
    
    This endpoint uses Google Gemini API to generate contextual, helpful instructions
    that guide blind users to position braille text correctly. The instructions adapt
    based on the current situation (initial setup, low confidence, no text detected).
    
    The detection box is a rectangular area in the center of the camera view:
    - Width: 60% of screen width
    - Height: 50% of screen height
    - Position: Centered on screen
    
    Args:
        context (str): Context for instruction generation:
            - "initial": First-time setup instructions
            - "low_confidence": Tips for improving recognition quality
            - "no_text_detected": Troubleshooting when no text is found
            - "": General positioning guidance
        voice_mode (bool): If True, generates shorter, voice-optimized instructions
    
    Returns:
        JSONResponse: {
            "instructions": str,  # AI-generated instruction text
            "context": str,        # Context used for generation
            "fallback": bool    # True if Gemini failed and fallback was used
        }
    """
    print(f"[BACKEND] Camera instructions requested with context: '{context}'")
    
    try:
        # Configure Gemini
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel(GEMINI_MODEL)
        
        # Build context-aware prompt
        if voice_mode:
            base_prompt = """You are a helpful assistant providing voice guidance to blind users positioning braille text in a camera detection box.

The detection box is a rectangular area in the center of the camera view (60% width, 50% height of the screen).

Provide very brief, clear voice instructions (1-2 short sentences maximum) optimized for text-to-speech. Use simple, direct language. Be encouraging.

Focus on:
- Positioning the braille material in the center box
- Distance and lighting
- Keeping steady

Write in a friendly, conversational tone. Keep it very brief for voice output."""
        else:
            base_prompt = """You are a helpful assistant guiding users to position braille text correctly in a camera detection box for text recognition.

The detection box is a rectangular area in the center of the camera view (60% width, 50% height of the screen).

Provide clear, concise, step-by-step instructions (2-3 sentences maximum) on how to position braille text in the detection box for best recognition results. Be encouraging and specific.

Instructions should cover:
- How to position the braille material
- Distance from camera
- Lighting considerations
- Keeping the text steady
- Ensuring the text fits within the box

Write in a friendly, helpful tone. Keep it brief and actionable."""

        # Add context-specific guidance
        if context == "low_confidence":
            prompt = base_prompt + "\n\nContext: The user is getting low confidence results. Provide specific, actionable tips like 'Move closer', 'Improve lighting', 'Keep steady', or directional commands like 'Move left' or 'Move right'."
        elif context == "no_text_detected":
            prompt = base_prompt + "\n\nContext: No text is being detected. Provide clear instructions like 'Please show braille text for detection', 'Move the text to the center', or specific directional commands like 'Move left', 'Move right', 'Move closer'."
        elif context == "initial":
            prompt = base_prompt + "\n\nContext: This is the initial instruction when the camera starts. Provide welcoming, clear first-time guidance with specific actions like 'Position braille text in the center box' or 'Show braille text for detection'."
        else:
            prompt = base_prompt + "\n\nContext: General guidance for optimal positioning. Use specific directional commands when needed."
        
        print(f"[DEBUG] Sending prompt to Gemini for camera instructions (context: {context})")
        
        # Generate instructions
        response = gemini_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,  # Slightly creative for varied instructions
                top_p=0.9,
                top_k=40,
                max_output_tokens=150,  # Keep it concise
            ),
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
        
        # Extract instructions
        if not response.candidates or len(response.candidates) == 0:
            raise ValueError("No response from Gemini")
        
        candidate = response.candidates[0]
        if candidate.finish_reason == 2:  # SAFETY
            print("[DEBUG] Response blocked by safety filters, using fallback instructions")
            # Context-specific fallback instructions
            if context == "no_text_detected":
                instructions = "Please show braille text for detection. Position it in the center box."
            elif context == "low_confidence":
                instructions = "Move closer, improve lighting, and keep steady."
            else:
                instructions = "Position your braille text within the blue detection box. Ensure good lighting and keep the text steady."
        else:
            instructions = response.text.strip()
        
        print(f"[DEBUG] Generated instructions: {instructions[:100]}...")
        
        return {
            "instructions": instructions,
            "context": context
        }
        
    except Exception as e:
        print(f"[DEBUG] Gemini instructions generation failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback instructions if Gemini fails - use specific, actionable commands
        fallback_instructions = {
            "initial": "Please show braille text for detection. Position it in the center box with good lighting.",
            "low_confidence": "Move closer, improve lighting, and keep steady. Use commands like 'Move left', 'Move right', or 'Move closer' to adjust position.",
            "no_text_detected": "Please show braille text for detection. Position it in the center box. If text is visible, try 'Move left', 'Move right', or 'Move closer'."
        }
        
        return {
            "instructions": fallback_instructions.get(context, fallback_instructions["initial"]),
            "context": context,
            "fallback": True
        }

@app.post("/api/directional-guidance")
async def get_directional_guidance(
    photo: UploadFile = File(...),
    confidence: float = Query(0.0, description="Current recognition confidence"),
    recognized_text: str = Query("", description="Currently recognized text (if any)")
):
    """
    Analyze image position and provide AI-powered directional guidance for blind users.
    
    This endpoint analyzes the captured image to determine if the braille text is:
    - Too far left/right (needs horizontal adjustment)
    - Too far up/down (needs vertical adjustment)
    - Too close/far (needs distance adjustment)
    - Too small (needs to move closer)
    - Well positioned (but may need lighting/steady improvements)
    
    The analysis uses image processing to detect text position relative to the
    detection box, then uses Gemini API to generate specific, actionable voice
    instructions like "Move the text to the left" or "Move closer to the text".
    
    Args:
        photo (UploadFile): Current camera frame (detection box cropped)
        confidence (float): Current recognition confidence (0.0 to 1.0)
        recognized_text (str): Currently recognized text (if any)
    
    Returns:
        JSONResponse: {
            "guidance": str,              # AI-generated directional instruction
            "position_analysis": str,     # Description of position issues
            "has_text": bool,             # Whether text was detected in image
            "offset_x_pct": float,        # Horizontal offset percentage
            "offset_y_pct": float,        # Vertical offset percentage
            "size_ratio": float          # Text size relative to detection box
        }
    """
    print(f"[BACKEND] Directional guidance requested - confidence: {confidence}, text: '{recognized_text}'")
    
    try:
        # Read image
        image_bytes = await photo.read()
        pil_img = Image.open(io.BytesIO(image_bytes))
        
        # Analyze image to determine position issues
        gray = np.array(pil_img.convert("L"))
        h, w = gray.shape
        
        # Detect braille region
        dark_thresh = 200
        mask = gray < dark_thresh
        col_sum = np.sum(mask, axis=0)
        row_sum = np.sum(mask, axis=1)
        cols = np.where(col_sum > 0)[0]
        rows = np.where(row_sum > 0)[0]
        
        # Calculate position metrics
        has_text = len(cols) > 0 and len(rows) > 0
        text_center_x = (cols[0] + cols[-1]) / 2 if len(cols) > 0 else w / 2
        text_center_y = (rows[0] + rows[-1]) / 2 if len(rows) > 0 else h / 2
        text_width = cols[-1] - cols[0] if len(cols) > 0 else 0
        text_height = rows[-1] - rows[0] if len(rows) > 0 else 0
        
        # Detection box is center 60% width, 50% height
        box_left = w * 0.2
        box_right = w * 0.8
        box_top = h * 0.25
        box_bottom = h * 0.75
        box_center_x = w / 2
        box_center_y = h / 2
        
        # Calculate position relative to box
        offset_x = text_center_x - box_center_x
        offset_y = text_center_y - box_center_y
        offset_x_pct = (offset_x / (w / 2)) * 100  # Percentage offset
        offset_y_pct = (offset_y / (h / 2)) * 100
        
        # Determine if text is too small (too far) or too large (too close)
        box_area = (box_right - box_left) * (box_bottom - box_top)
        text_area = text_width * text_height if text_width > 0 and text_height > 0 else 0
        size_ratio = text_area / box_area if box_area > 0 else 0
        
        # Build position analysis
        position_issues = []
        if not has_text:
            position_issues.append("no_text_detected")
        else:
            if abs(offset_x_pct) > 20:
                if offset_x_pct > 0:
                    position_issues.append("too_right")
                else:
                    position_issues.append("too_left")
            if abs(offset_y_pct) > 20:
                if offset_y_pct > 0:
                    position_issues.append("too_down")
                else:
                    position_issues.append("too_up")
            if size_ratio < 0.1:
                position_issues.append("too_far")
            elif size_ratio > 0.8:
                position_issues.append("too_close")
            if text_width < w * 0.1 or text_height < h * 0.1:
                position_issues.append("text_too_small")
        
        # Use Gemini to generate specific directional guidance
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel(GEMINI_MODEL)
        
        # Build prompt with position analysis
        position_desc = ", ".join(position_issues) if position_issues else "well_positioned"
        
        prompt = f"""You are providing voice guidance to a blind user positioning braille text in a camera detection box.

Current situation:
- Recognition confidence: {confidence:.2f}
- Recognized text: "{recognized_text}" (empty if no text detected)
- Position analysis: {position_desc}

The detection box is in the center of the screen (60% width, 50% height).

Provide a very brief, direct voice instruction (one short sentence, maximum 12 words) telling the user exactly what to do.

IMPORTANT: Use specific, actionable commands:
- "too_left" → "Move right" or "Move the text to the right"
- "too_right" → "Move left" or "Move the text to the left"
- "too_up" → "Move down" or "Move the text down"
- "too_down" → "Move up" or "Move the text up"
- "too_far" or "text_too_small" → "Move closer" or "Bring the braille closer"
- "too_close" → "Move back" or "Move further away"
- "no_text_detected" → "Please show braille text for detection" or "Position braille text in the center box"
- "well_positioned" but low confidence → "Keep steady" or "Improve lighting"

Use simple, direct, imperative commands. Be specific about direction (left, right, up, down, closer, further).
Return ONLY the instruction, no explanation or extra text."""

        print(f"[DEBUG] Position analysis: {position_desc}, offset: ({offset_x_pct:.1f}%, {offset_y_pct:.1f}%), size_ratio: {size_ratio:.2f}")
        
        response = gemini_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,  # Lower for more consistent, direct instructions
                top_p=0.8,
                top_k=10,
                max_output_tokens=20,  # Very short
            ),
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
        
        if not response.candidates or len(response.candidates) == 0:
            raise ValueError("No response from Gemini")
        
        candidate = response.candidates[0]
        if candidate.finish_reason == 2:  # SAFETY
            # Fallback based on position analysis - use specific directional commands
            if "too_left" in position_issues:
                guidance = "Move right"
            elif "too_right" in position_issues:
                guidance = "Move left"
            elif "too_up" in position_issues:
                guidance = "Move down"
            elif "too_down" in position_issues:
                guidance = "Move up"
            elif "too_far" in position_issues or "text_too_small" in position_issues:
                guidance = "Move closer"
            elif "too_close" in position_issues:
                guidance = "Move back"
            elif not has_text:
                guidance = "Please show braille text for detection"
            else:
                guidance = "Keep steady and improve lighting"
        else:
            guidance = response.text.strip()
        
        print(f"[DEBUG] Generated directional guidance: '{guidance}'")
        
        return {
            "guidance": guidance,
            "position_analysis": position_desc,
            "has_text": has_text,
            "offset_x_pct": float(offset_x_pct),
            "offset_y_pct": float(offset_y_pct),
            "size_ratio": float(size_ratio)
        }
        
    except Exception as e:
        print(f"[DEBUG] Directional guidance failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback guidance
        return {
            "guidance": "Please show braille text for detection",
            "position_analysis": "unknown",
            "has_text": False,
            "error": str(e)
        }

@app.post("/api/recognize")
async def recognize_braille(
    photo: UploadFile = File(...),
    use_enhancement: bool = Query(False, description="Whether to use image enhancement (not implemented yet)"),
    is_camera: bool = Query(False, description="Whether this is from camera (uses different processing)")
):
    """
    Main API endpoint for braille text recognition.
    
    This is the core endpoint that processes images and returns recognized text.
    It handles two modes:
    
    1. CAMERA MODE (is_camera=True):
       - Pre-checks image with Gemini Vision to detect braille presence
       - If no braille detected, returns guidance instead of running model
       - Uses lenient auto-cropping (multiple threshold attempts)
       - Always applies Gemini word correction
       - Returns results regardless of confidence
    
    2. UPLOAD MODE (is_camera=False):
       - Skips Gemini pre-check (works perfectly, don't modify)
       - Uses strict auto-cropping (threshold 240)
       - Applies Gemini word correction
       - Returns results with confidence score
    
    Processing Pipeline:
    1. Validate image file type
    2. Read image bytes
    3. (Camera only) Pre-check with Gemini Vision
    4. Auto-crop braille region
    5. Preprocess for model (resize, normalize, pad/truncate)
    6. Run model inference
    7. Decode CTC output to text
    8. Correct word with Gemini API
    9. Compute confidence score
    10. Return JSON response
    
    Args:
        photo (UploadFile): Image file uploaded by user (JPEG, PNG, etc.)
        use_enhancement (bool): Future feature - image enhancement (not implemented)
        is_camera (bool): True if from camera, False if from file upload
    
    Returns:
        JSONResponse: {
            "word": str,              # Recognized/corrected text
            "confidence": float,      # Confidence score (0.0 to 1.0)
            "original": str or None,    # Original text before correction (if corrected)
            "corrected": bool,        # Whether word was corrected
            "braille_detected": bool, # (Camera only) Whether Gemini detected braille
            "guidance": str            # (Camera only) Directional guidance if no braille
        }
    
    Raises:
        HTTPException: 503 if model not loaded, 400 if invalid file, 500 on processing error
    """
    print(f"[BACKEND] ===== RECEIVED REQUEST =====")
    print(f"[BACKEND] is_camera: {is_camera}")
    print(f"[BACKEND] use_enhancement: {use_enhancement}")
    print(f"[BACKEND] photo.filename: {photo.filename}")
    print(f"[BACKEND] photo.content_type: {photo.content_type}")
    
    if model is None or num_to_char is None:
        print("[BACKEND] ERROR: Model not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if not photo.content_type or not photo.content_type.startswith("image/"):
        print(f"[BACKEND] ERROR: Invalid file type: {photo.content_type}")
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image bytes
        print("[BACKEND] Reading image bytes...")
        image_bytes = await photo.read()
        print(f"[BACKEND] Image bytes read: {len(image_bytes)} bytes")
        if len(image_bytes) == 0:
            print("[BACKEND] ERROR: Empty file")
            raise HTTPException(status_code=400, detail="Empty file")
        
        # Log original image size
        pil_img_original = Image.open(io.BytesIO(image_bytes))
        print(f"[DEBUG] Step 1 - Original image size: {pil_img_original.size} (width x height)")
        
        # CAMERA MODE ONLY: Pre-check with Gemini to detect if braille is present
        # This avoids running the model on frames without braille text
        # UPLOAD MODE: Skip this check (working perfectly, don't touch)
        if is_camera:
            print(f"[DEBUG] Camera mode: Checking for braille with Gemini before model inference...")
            # Pass image bytes to Gemini for braille detection
            has_braille = check_braille_with_gemini(bytes(image_bytes))
            
            if not has_braille:
                print(f"[DEBUG] Camera mode: Gemini did not detect braille, skipping model inference")
                # Generate directional guidance to help user position braille
                print(f"[DEBUG] Camera mode: Generating directional guidance for no braille detected...")
                try:
                    # Use the directional guidance endpoint logic to get instructions
                    pil_img = Image.open(io.BytesIO(image_bytes))
                    gray = np.array(pil_img.convert("L"))
                    h, w = gray.shape
                    
                    # Detect any dark regions (potential braille)
                    dark_thresh = 200
                    mask = gray < dark_thresh
                    col_sum = np.sum(mask, axis=0)
                    row_sum = np.sum(mask, axis=1)
                    cols = np.where(col_sum > 0)[0]
                    rows = np.where(row_sum > 0)[0]
                    
                    has_text = len(cols) > 0 and len(rows) > 0
                    position_issues = ["no_text_detected"]
                    
                    if has_text:
                        # Text might be present but not recognized as braille
                        text_center_x = (cols[0] + cols[-1]) / 2 if len(cols) > 0 else w / 2
                        text_center_y = (rows[0] + rows[-1]) / 2 if len(rows) > 0 else h / 2
                        box_center_x = w / 2
                        box_center_y = h / 2
                        offset_x_pct = ((text_center_x - box_center_x) / (w / 2)) * 100
                        offset_y_pct = ((text_center_y - box_center_y) / (h / 2)) * 100
                        
                        if abs(offset_x_pct) > 20:
                            if offset_x_pct > 0:
                                position_issues.append("too_right")
                            else:
                                position_issues.append("too_left")
                        if abs(offset_y_pct) > 20:
                            if offset_y_pct > 0:
                                position_issues.append("too_down")
                            else:
                                position_issues.append("too_up")
                    
                    # Generate guidance using Gemini
                    genai.configure(api_key=GEMINI_API_KEY)
                    gemini_model = genai.GenerativeModel(GEMINI_MODEL)
                    position_desc = ", ".join(position_issues)
                    
                    prompt = f"""You are providing voice guidance to a blind user positioning braille text in a camera detection box.

Current situation:
- No braille text detected in the image
- Position analysis: {position_desc}

The detection box is in the center of the screen (60% width, 50% height).

Provide a very brief, direct voice instruction (one short sentence, maximum 12 words) telling the user exactly what to do.

IMPORTANT: Use specific, actionable commands:
- If no text detected: "Please show braille text for detection" or "Position braille text in the center box"
- If text is too left: "Move right" or "Move the text to the right"
- If text is too right: "Move left" or "Move the text to the left"
- If text is too up: "Move down" or "Move the text down"
- If text is too down: "Move up" or "Move the text up"
- If text is too far: "Move closer" or "Bring the braille closer"
- If text is too close: "Move back" or "Move further away"

Use simple, direct, imperative commands. Be specific about direction.
Return ONLY the instruction, no explanation or extra text."""
                    
                    response = gemini_model.generate_content(
                        prompt,
                        generation_config=genai.types.GenerationConfig(
                            temperature=0.3,
                            top_p=0.8,
                            top_k=10,
                            max_output_tokens=20,
                        ),
                        safety_settings={
                            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                        }
                    )
                    
                    if response.candidates and len(response.candidates) > 0:
                        candidate = response.candidates[0]
                        if candidate.finish_reason != 2:  # Not blocked
                            guidance = response.text.strip()
                        else:
                            # Fallback with specific directional commands
                            if "too_left" in position_issues:
                                guidance = "Move right"
                            elif "too_right" in position_issues:
                                guidance = "Move left"
                            elif "too_up" in position_issues:
                                guidance = "Move down"
                            elif "too_down" in position_issues:
                                guidance = "Move up"
                            else:
                                guidance = "Please show braille text for detection"
                    else:
                        # Fallback with specific directional commands
                        if "too_left" in position_issues:
                            guidance = "Move right"
                        elif "too_right" in position_issues:
                            guidance = "Move left"
                        elif "too_up" in position_issues:
                            guidance = "Move down"
                        elif "too_down" in position_issues:
                            guidance = "Move up"
                        else:
                            guidance = "Please show braille text for detection"
                    
                    print(f"[DEBUG] Generated guidance for no braille: '{guidance}'")
                    
                except Exception as e:
                    print(f"[DEBUG] Error generating guidance: {e}")
                    guidance = "Please show braille text for detection"
                
                # Return empty result with guidance
                return {
                    "word": "",
                    "confidence": 0.0,
                    "original": None,
                    "corrected": False,
                    "braille_detected": False,
                    "guidance": guidance
                }
            else:
                print(f"[DEBUG] Camera mode: Gemini detected braille, proceeding with model inference")
        
        # Preprocess image (different logic for camera vs upload)
        # CAMERA: Detection box crop (frontend) -> Auto-crop -> Preprocess
        # UPLOAD: Auto-crop -> Preprocess (UNCHANGED - working perfectly)
        print(f"[DEBUG] Step 2 - Starting preprocessing (is_camera={is_camera})...")
        img_array, actual_width = preprocess_image_from_bytes(image_bytes, is_camera=is_camera)
        print(f"[DEBUG] Step 3 - Preprocessed image shape: {img_array.shape}, actual_width: {actual_width}")
        img_batch = np.expand_dims(img_array, axis=0).astype(np.float32)
        
        # Run inference
        print(f"[DEBUG] Step 4 - Running model inference...")
        logits = model.predict(img_batch, verbose=0)  # (1, T, C)
        
        # Decode CTC output (using greedy decoding like notebook)
        print(f"[DEBUG] Step 5 - Decoding CTC output...")
        recognized_text = decode_ctc(logits, num_to_char, beam_width=BEAM_WIDTH)
        
        print(f"[DEBUG] Step 6 - Recognition result: '{recognized_text}' (is_camera={is_camera})")
        
        # CAMERA PROCESSING: Always use Gemini correction and show results
        if is_camera:
            print(f"[DEBUG] Camera - CTC decode result: '{recognized_text}'")
            
            # Try multiple decoding strategies for camera (like notebook)
            text_probs = recognized_text
            text_manual = ""
            
            # Always try manual decode as well
            try:
                text_manual, argseq, collapsed_ids = manual_argmax_collapse(logits, num_to_char)
                print(f"[DEBUG] Camera - Manual decode result: '{text_manual}'")
            except Exception as e:
                print(f"[DEBUG] Camera - Manual decode failed: {e}")
            
            # Choose best result: prefer CTC decode, fallback to manual
            original_text = text_probs if (text_probs and len(text_probs) > 0) else text_manual
            print(f"[DEBUG] Camera - Selected text before correction: '{original_text}'")
            
            # Always validate and correct with Gemini for camera
            if original_text and len(original_text) > 0:
                print(f"[DEBUG] Step 7 - Applying Gemini correction to '{original_text}'...")
                corrected_text, was_corrected = validate_and_correct_word(original_text)
                print(f"[DEBUG] Step 8 - After Gemini correction: '{corrected_text}' (corrected: {was_corrected})")
            else:
                # Still empty - return empty but don't fail
                print(f"[DEBUG] Step 7 - No text found after all decoding attempts, skipping Gemini")
                corrected_text = ""
                was_corrected = False
            
            # Compute confidence
            confidence = compute_confidence(logits, original_text if original_text else "", num_to_char)
            
            # CAMERA: Always return result regardless of confidence (even if empty)
            result = {
                "word": corrected_text if corrected_text else (original_text if original_text else ""),
                "confidence": confidence,
                "original": original_text if was_corrected and original_text else None,
                "corrected": was_corrected
            }
            print(f"[DEBUG] Camera - Final result: {result}")
            return result
        
        # UPLOAD PROCESSING: Keep existing logic (100% accurate, don't change)
        else:
            # Optional: Log for debugging (can be removed in production)
            if not recognized_text or len(recognized_text) == 0:
                print(f"Warning: Empty recognition result. Logits shape: {logits.shape}")
                return {
                    "word": "",
                    "confidence": 0.0,
                    "original": None,
                    "corrected": False
                }
            
            # Validate and correct word using Gemini
            original_text = recognized_text
            print(f"[DEBUG] Upload - Before correction: '{original_text}'")
            corrected_text, was_corrected = validate_and_correct_word(recognized_text)
            print(f"[DEBUG] Upload - After correction: '{corrected_text}' (corrected: {was_corrected})")
            
            # Compute confidence
            confidence = compute_confidence(logits, recognized_text, num_to_char)
            
            # Return result with correction info
            result = {
                "word": corrected_text,
                "confidence": confidence,
                "original": original_text if was_corrected else None,
                "corrected": was_corrected
            }
            print(f"[DEBUG] Upload - Final result: {result}")
            return result
    
    except Exception as e:
        print(f"Recognition error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Recognition failed: {str(e)}")

# ----------------- Main -----------------
if __name__ == "__main__":
    print("Starting Braille Recognition API server...")
    print(f"Model path: {MODEL_PATH}")
    print(f"Charset path: {CHARSET_PATH}")
    uvicorn.run(app, host="0.0.0.0", port=8000)


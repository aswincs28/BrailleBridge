"""
Standalone test harness for braille recognition pipeline.

Loads the same model and charset as main.py, then runs each test image
through several preprocessing variants and reports the decoded text.

Goal: figure out which preprocessing strategy reproduces the trained-model
inference correctly on the user's WhatsApp images.
"""
import os
import sys
import io
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "best_ctc_inference_2.keras")
CHARSET_PATH = os.path.join(HERE, "charset.txt")
IMG_HEIGHT = 64
MAX_IMG_WIDTH = 640

# Locate test images: parent NEW2 folder
NEW2_DIR = os.path.dirname(HERE)
TEST_IMAGES = sorted([
    os.path.join(NEW2_DIR, f) for f in os.listdir(NEW2_DIR)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
])

# Expected outputs (user said: acid, hello, name in some order)
EXPECTED_SET = {"ACID", "HELLO", "NAME"}


def load_charset(path):
    toks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.rstrip("\n\r")
            if s:
                toks.append(s)
    return {i: c for i, c in enumerate(toks)}, toks


# ---------- Preprocessing variants ----------

def preprocess_notebook_style(pil_img, img_height=IMG_HEIGHT, max_width=MAX_IMG_WIDTH):
    """Exact preprocessing from training notebook: resize + pad. NO crop."""
    img = pil_img.convert("L")
    w, h = img.size
    new_h = img_height
    new_w = max(1, int(w * (new_h / float(h))))
    img = img.resize((new_w, new_h), Image.LANCZOS)
    arr = np.array(img).astype(np.float32) / 255.0
    arr = arr * 2.0 - 1.0
    if new_w < max_width:
        pad = np.ones((new_h, max_width), dtype=np.float32) * (-1.0)
        pad[:, :new_w] = arr
        arr = pad
    elif new_w > max_width:
        arr = arr[:, :max_width]
        new_w = max_width
    arr = np.expand_dims(arr, -1)
    return arr, new_w


def autocrop_original(pil_img, dark_thresh=240, pad=10):
    """Backend's current autocrop (before any fix)."""
    gray = np.array(pil_img.convert("L"))
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    mask = gray < dark_thresh
    cols = np.where(np.sum(mask, axis=0) > 0)[0]
    rows = np.where(np.sum(mask, axis=1) > 0)[0]
    if len(cols) == 0 or len(rows) == 0:
        return pil_img.copy()
    x1, x2 = int(cols[0]), int(cols[-1])
    y1, y2 = int(rows[0]), int(rows[-1])
    h, w = gray.shape
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad); y2 = min(h, y2 + pad)
    return pil_img.crop((x1, y1, x2, y2))


def autocrop_smart(pil_img, dark_thresh=200, pad=10):
    """
    Smart autocrop that drops lone-bottom-row punctuation dots at the edges.

    Strategy:
    1. Find all dark connected components (after morphology open).
    2. Drop tiny noise blobs (area < 0.2 * median).
    3. Detect bottom-row Y level (the highest cy among kept components).
    4. For each edge (leftmost and rightmost component): if it sits in the bottom row
       AND no other component shares its X-column within ~half a dot-spacing, drop it.
       Such a dot is almost always a stray braille punctuation cell (dot 3 / dot 6 only)
       that the A-Z-only model would hallucinate as a phantom letter.
    5. Crop to the bounding box of surviving components.
    """
    gray = np.array(pil_img.convert("L"))
    h_img, w_img = gray.shape
    binary = (gray < dark_thresh).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    n_labels, _, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n_labels <= 1:
        return pil_img.copy()

    comps = []
    for i in range(1, n_labels):
        x, y, w, h, area = stats[i]
        cx, cy = centroids[i]
        comps.append({"x": x, "y": y, "w": w, "h": h, "area": area, "cx": cx, "cy": cy})
    if not comps:
        return pil_img.copy()

    areas = sorted(c["area"] for c in comps)
    median_area = areas[len(areas) // 2]
    comps = [c for c in comps if c["area"] >= max(4, 0.2 * median_area)]
    if not comps:
        return pil_img.copy()

    # Estimate row Y levels: top is min cy, bottom is max cy
    cys = [c["cy"] for c in comps]
    top_y, bot_y = min(cys), max(cys)
    row_span = bot_y - top_y
    # If row span is tiny (single row), there's nothing to filter
    if row_span < 5:
        keep = comps
    else:
        # Bottom-row threshold: cy >= bot_y - 25% of row span
        bot_threshold = bot_y - 0.25 * row_span
        # Estimate intra-cell column spacing (median of nearest-neighbor X gaps)
        sorted_cx = sorted(set(round(c["cx"]) for c in comps))
        gaps = [b - a for a, b in zip(sorted_cx[:-1], sorted_cx[1:])]
        # Cell column spacing ~ smallest gap (intra-cell L-R column distance)
        col_tol = (min(gaps) if gaps else 30) * 0.7

        def has_column_companion(target):
            for c in comps:
                if c is target:
                    continue
                if abs(c["cx"] - target["cx"]) <= col_tol:
                    return True
            return False

        keep = list(comps)
        # Strip leftmost lone-bottom-dot
        keep_sorted_left = sorted(keep, key=lambda c: c["cx"])
        leftmost = keep_sorted_left[0]
        if leftmost["cy"] >= bot_threshold and not has_column_companion(leftmost):
            keep.remove(leftmost)
        # Strip rightmost lone-bottom-dot
        if keep:
            keep_sorted_right = sorted(keep, key=lambda c: -c["cx"])
            rightmost = keep_sorted_right[0]
            if rightmost["cy"] >= bot_threshold and not has_column_companion(rightmost):
                keep.remove(rightmost)
        if not keep:
            keep = comps  # fallback

    x1 = min(c["x"] for c in keep)
    y1 = min(c["y"] for c in keep)
    x2 = max(c["x"] + c["w"] for c in keep)
    y2 = max(c["y"] + c["h"] for c in keep)
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(w_img, x2 + pad); y2 = min(h_img, y2 + pad)
    return pil_img.crop((x1, y1, x2, y2))


def autocrop_robust(pil_img, dark_thresh=200, pad=10):
    """Improved autocrop: morphology + count threshold to ignore noise."""
    gray = np.array(pil_img.convert("L"))
    h_img, w_img = gray.shape
    m = (gray < dark_thresh).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
    col_min = max(2, int(0.01 * h_img))
    row_min = max(2, int(0.01 * w_img))
    cols = np.where(np.sum(m, axis=0) >= col_min)[0]
    rows = np.where(np.sum(m, axis=1) >= row_min)[0]
    if len(cols) == 0 or len(rows) == 0:
        return pil_img.copy()
    x1, x2 = int(cols[0]), int(cols[-1])
    y1, y2 = int(rows[0]), int(rows[-1])
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(w_img, x2 + pad); y2 = min(h_img, y2 + pad)
    return pil_img.crop((x1, y1, x2, y2))


def add_whitespace_padding(pil_img, pad_ratio=0.1):
    """Add white margin around image to give CTC time-axis breathing room."""
    w, h = pil_img.size
    pad_x = max(8, int(w * pad_ratio))
    pad_y = max(4, int(h * pad_ratio))
    new_w = w + 2 * pad_x
    new_h = h + 2 * pad_y
    canvas = Image.new("L", (new_w, new_h), color=255)
    canvas.paste(pil_img.convert("L"), (pad_x, pad_y))
    return canvas


# ---------- CTC decoding ----------

def decode_ctc(logits, num_to_char, beam_width=10):
    """Match notebook decode_batch."""
    probs = tf.nn.softmax(logits, axis=-1).numpy()
    batch_size = probs.shape[0]
    time_steps = probs.shape[1]
    input_length = np.ones((batch_size,), dtype=np.int32) * time_steps
    decoded, _ = tf.keras.backend.ctc_decode(
        probs, input_length=input_length,
        greedy=(beam_width == 1), beam_width=beam_width, top_paths=1
    )
    first = decoded[0]
    if isinstance(first, tf.SparseTensor):
        dense = tf.sparse.to_dense(first, default_value=-1).numpy()
    elif tf.is_tensor(first):
        dense = first.numpy()
    else:
        dense = np.asarray(first)
    blank = logits.shape[-1] - 1
    out_texts = []
    for row in dense:
        chars = []
        for cid in row:
            cid = int(cid)
            if cid == -1 or cid >= blank:
                continue
            chars.append(num_to_char.get(cid, "?"))
        out_texts.append("".join(chars))
    return out_texts[0] if out_texts else ""


def manual_argmax(logits, num_to_char):
    arg = np.argmax(logits, axis=-1)[0].tolist()
    blank = logits.shape[-1] - 1
    out, prev = [], None
    for a in arg:
        if a != prev and a != blank:
            out.append(a)
        prev = a
    return "".join(num_to_char.get(int(i), "?") for i in out)


# ---------- Test harness ----------

def run_variants(model, num_to_char, image_path):
    pil_orig = Image.open(image_path)
    print(f"\n{'='*70}")
    print(f"IMAGE: {os.path.basename(image_path)}  size={pil_orig.size}")
    print('='*70)

    # G: Pad vertically to a target aspect ratio. Braille cell ≈ 2 wide × 3 tall dots.
    # A word of N cells has aspect width/height ≈ 2N/3 ≈ 0.67N. With dot spacing,
    # natural per-cell aspect (with margin) is closer to width/height ≈ 1.0 per cell.
    # So target ratio = ~1.0 * num_cells. We don't know num_cells, but capping the
    # ratio so cropped braille is at most ~5x wider than tall covers most words.
    def pad_to_target_aspect(pil, target_aspect=4.0):
        w, h = pil.size
        cur = w / max(1, h)
        if cur <= target_aspect:
            return pil  # already tall enough
        new_h = int(w / target_aspect)
        canvas = Image.new("L", (w, new_h), color=255)
        y_off = (new_h - h) // 2
        canvas.paste(pil.convert("L"), (0, y_off))
        return canvas

    cropped_robust = autocrop_robust(pil_orig, dark_thresh=200, pad=10)
    cropped_smart  = autocrop_smart(pil_orig, dark_thresh=200, pad=10)
    variants = {
        "H_robust_aspect_3.0    ": pad_to_target_aspect(cropped_robust, 3.0),
        "P_smart_raw            ": cropped_smart,
        "Q_smart_aspect_3.0     ": pad_to_target_aspect(cropped_smart, 3.0),
        "R_smart_aspect_2.5     ": pad_to_target_aspect(cropped_smart, 2.5),
        "S_smart_aspect_2.0     ": pad_to_target_aspect(cropped_smart, 2.0),
        "T_smart_aspect_4.0     ": pad_to_target_aspect(cropped_smart, 4.0),
    }

    for name, pil in variants.items():
        try:
            arr, actual_w = preprocess_notebook_style(pil)
            X = np.expand_dims(arr, 0)  # (1, H, W, 1)
            logits = model.predict(X, verbose=0)
            ctc_b10 = decode_ctc(logits, num_to_char, beam_width=10)
            ctc_b1  = decode_ctc(logits, num_to_char, beam_width=1)
            argmax  = manual_argmax(logits, num_to_char)
            cropped_size = pil.size
            print(f"  {name} | crop={cropped_size!s:>14} actual_w={actual_w:>3} | beam10='{ctc_b10}' | greedy='{ctc_b1}' | argmax='{argmax}'")
        except Exception as e:
            print(f"  {name} | ERROR: {type(e).__name__}: {e}")


def main():
    print(f"Loading charset from {CHARSET_PATH}")
    num_to_char, toks = load_charset(CHARSET_PATH)
    print(f"  Charset: {toks}")
    print(f"  Vocab size (with blank): {len(toks) + 1}")

    print(f"\nLoading model from {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print(f"  Model input shape: {model.input_shape}")
    print(f"  Model output shape: {model.output_shape}")

    print(f"\nFound {len(TEST_IMAGES)} test images:")
    for p in TEST_IMAGES:
        print(f"  - {p}")

    for img_path in TEST_IMAGES:
        run_variants(model, num_to_char, img_path)

    print(f"\n{'='*70}")
    print(f"Expected ground truths (in some order): {sorted(EXPECTED_SET)}")
    print('='*70)


if __name__ == "__main__":
    main()

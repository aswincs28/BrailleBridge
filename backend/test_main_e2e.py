"""
End-to-end regression test: import main.py's preprocessing + decoding
and verify the 3 ground-truth images produce ACID/HELLO/NAME.

Run: .venv/Scripts/python.exe test_main_e2e.py
Exit code 0 = all pass.
"""
import os
import sys
import tensorflow as tf
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import main  # noqa: E402

NEW2_DIR = os.path.dirname(HERE)

# Map filename -> expected uppercase ground truth
GT = {
    "WhatsApp Image 2026-05-04 at 10.16.03 PM.jpeg": "HELLO",
    "WhatsApp Image 2026-05-04 at 10.16.48 PM.jpeg": "NAME",
    "WhatsApp Image 2026-05-04 at 10.18.25 PM.jpeg": "ACID",
}


def main_test():
    num_to_char, _ = main.load_charset(main.CHARSET_PATH)
    model = tf.keras.models.load_model(main.MODEL_PATH, compile=False)
    print(f"Model loaded. Input {model.input_shape} -> output {model.output_shape}")

    failures = []
    for fname, expected in GT.items():
        path = os.path.join(NEW2_DIR, fname)
        with open(path, "rb") as f:
            image_bytes = f.read()
        arr, actual_w = main.preprocess_image_from_bytes(image_bytes, is_camera=False)
        X = np.expand_dims(arr, 0)
        logits = model.predict(X, verbose=0)
        pred = main.decode_ctc(logits, num_to_char, beam_width=main.BEAM_WIDTH)
        ok = pred.upper() == expected
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {fname[:40]:<40} expected={expected:<6} got={pred!r:<10} actual_w={actual_w}")
        if not ok:
            failures.append((fname, expected, pred))

    print()
    if failures:
        print(f"FAILED: {len(failures)}/{len(GT)} images")
        for f, e, p in failures:
            print(f"  - {f}: expected {e}, got {p!r}")
        return 1
    print(f"ALL {len(GT)} IMAGES PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main_test())

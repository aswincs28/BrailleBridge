import torch
import cv2
import numpy as np
from model.unet import UNetResNet34
from utils.decode import decode_braille


def load_model(weights="braille_unet.pth"):
    model = UNetResNet34(pretrained=False)
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    model.eval()
    return model


def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (256, 256))
    img = img / 255.0
    img = torch.tensor(img).unsqueeze(0).unsqueeze(0).float()
    return img


def predict_braille(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found")

    model = load_model()
    x = preprocess(img)

    with torch.no_grad():
        mask = model(x).squeeze().numpy()

    dots = (mask > 0.5).astype(np.uint8)
    text = decode_braille(dots)
    return text

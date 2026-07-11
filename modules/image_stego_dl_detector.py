import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import string
import re
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- RESNET18 MODEL (MUST MATCH TRAINING) ----------------
def build_model():
    model = models.resnet18(weights=None)   # no pretrained weights — we load our own

    # Same frozen structure as training (doesn't affect inference, just for consistency)
    model.fc = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, 2)
    )
    return model

# ---------------- LOAD MODEL ----------------
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pt")

try:
    model = build_model()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    print(f"✅ Stego model loaded from {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading stego model: {e}")
    model = None

# ---------------- TRANSFORM ----------------
# Must match test transform from training — 224x224, ImageNet normalize
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------- MESSAGE CLEAN ----------------
def clean_message(msg: str) -> str:
    msg = msg.strip()
    msg = re.sub(r'(.)\1{6,}', '', msg)
    msg = msg.rstrip("#").strip()
    return msg

def is_readable(msg: str) -> bool:
    if not msg:
        return False
    printable       = set(string.printable)
    printable_ratio = sum(c in printable for c in msg) / len(msg)
    alpha_ratio     = sum(c.isalpha() or c.isspace() for c in msg) / len(msg)
    return printable_ratio > 0.9 and alpha_ratio > 0.5

# ---------------- LSB EXTRACTION ----------------
def extract_message(image_path: str) -> str:
    img    = Image.open(image_path).convert("RGB")
    pixels = np.array(img)
    binary = ""

    for row in pixels:
        for pixel in row:
            r, g, b = pixel
            binary += str(r & 1)
            binary += str(g & 1)
            binary += str(b & 1)

    chars   = [binary[i:i+8] for i in range(0, len(binary), 8)]
    message = ""

    for c in chars:
        if len(c) < 8:
            break
        char     = chr(int(c, 2))
        message += char
        if message.endswith("###"):
            return message[:-3]

    return message

# ---------------- MAIN DETECTOR ----------------
def detect_stego_dl(image_path: str) -> str:
    if model is None:
        return "Error: Model not loaded. Please check models/best_model.pt exists."

    try:
        img        = Image.open(image_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output     = model(img_tensor)
            probs      = torch.softmax(output, dim=1)[0]
            _, pred    = torch.max(output, 1)
            pred_idx   = pred.item()
            confidence = probs[pred_idx].item() * 100

        # class 0 = cover, class 1 = stego
        if pred_idx == 1:
            raw_msg     = extract_message(image_path)
            cleaned_msg = clean_message(raw_msg)

            if not cleaned_msg:
                return (
                    f"Hidden data detected\n"
                    f"Confidence: {confidence:.2f}%\n"
                    f"No readable message found"
                )

            if is_readable(cleaned_msg):
                return (
                    f"Hidden data detected\n"
                    f"Confidence: {confidence:.2f}%\n"
                    f"Extracted Message: {cleaned_msg[:200]}"
                )

            preview = cleaned_msg[:60]
            return (
                f"Hidden data detected\n"
                f"Confidence: {confidence:.2f}%\n"
                f"Message not readable\n"
                f"Raw Data Preview: {preview}..."
            )

        else:
            return (
                f"No hidden data detected\n"
                f"Confidence: {confidence:.2f}%"
            )

    except Exception as e:
        return f"Error during analysis: {str(e)}"
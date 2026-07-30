import os
import torch

# --- HARDWARE & DEVICE CONFIGURATION ---
CAMERA_INDEX = int(os.getenv("NAVI_CAMERA_INDEX", 0))
CAMERA_RESOLUTION = (640, 480)
TARGET_FPS = 30

# Auto-detect CUDA GPU acceleration if available, fallback to CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- MODEL CONFIGURATION ---
YOLO_MODEL_PATH = os.getenv("NAVI_YOLO_MODEL", "yolo11n.pt")
CONFIDENCE_THRESHOLD = float(os.getenv("NAVI_CONF_THRESH", 0.45))
IOU_THRESHOLD = float(os.getenv("NAVI_IOU_THRESH", 0.45))

CUSTOM_MODEL_PATH = os.path.join(os.path.dirname(__file__), "custom_emotion_model.skops")

# --- AUDIO CONFIGURATION ---
AUDIO_RATE = int(os.getenv("NAVI_AUDIO_RATE", 160))
AUDIO_ENABLED = True

# --- SYSTEM CONFIGURATION ---
DEBUG_MODE = False
SMOOTHING_BUFFER_SIZE = 10
 

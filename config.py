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
AI_INPUT_SIZE = int(os.getenv("NAVI_AI_INPUT_SIZE", 640))

CUSTOM_MODEL_PATH = os.path.join(os.path.dirname(__file__), "custom_emotion_model.skops")

# --- AUDIO CONFIGURATION ---
AUDIO_RATE = int(os.getenv("NAVI_AUDIO_RATE", 160))
AUDIO_ENABLED = True

# --- SYSTEM CONFIGURATION ---
DEBUG_MODE = True
SMOOTHING_BUFFER_SIZE = 10

# --- NAVIGATION & DEPTH CONFIGURATION ---
DEPTH_MODEL = os.getenv("NAVI_DEPTH_MODEL", "MiDaS_small")
DANGER_DISTANCE = float(os.getenv("NAVI_DANGER_DIST", 0.7))    # Normalized relative depth (0.0 - 1.0) where 1.0 is closest
WARNING_DISTANCE = float(os.getenv("NAVI_WARNING_DIST", 0.5))
SPEECH_COOLDOWN = float(os.getenv("NAVI_SPEECH_COOLDOWN", 5.0)) # Seconds before repeating obstacle warning
NAVIGATION_REFRESH_RATE = float(os.getenv("NAVI_NAV_RATE", 2.0)) # Hz for navigation decisions

# --- THREAT DETECTION CONFIGURATION ---
TTC_CRITICAL_THRESHOLD = 1.5 # Seconds to collision
TTC_WARNING_THRESHOLD = 3.0 # Seconds to collision
HUMAN_PROX_VERY_CLOSE = 0.85 # Normalized depth
HUMAN_PROX_CLOSE = 0.7       # Normalized depth

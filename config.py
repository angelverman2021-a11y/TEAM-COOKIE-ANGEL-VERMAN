import os

# --- HARDWARE CONFIGURATION ---
CAMERA_INDEX = int(os.getenv("NAVI_CAMERA_INDEX", 0))

# --- MODEL CONFIGURATION ---
# Using the secure skops format instead of insecure pickle
CUSTOM_MODEL_PATH = os.path.join(os.path.dirname(__file__), "custom_emotion_model.skops")
YOLO_MODEL_PATH = "yolov8n.pt"

# --- AUDIO CONFIGURATION ---
AUDIO_RATE = int(os.getenv("NAVI_AUDIO_RATE", 160))
AUDIO_ENABLED = True

# --- SYSTEM CONFIGURATION ---
DEBUG_MODE = False
SMOOTHING_BUFFER_SIZE = 10 

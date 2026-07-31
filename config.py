import os
import torch

# --- HARDWARE & DEVICE CONFIGURATION ---
CAMERA_INDEX = int(os.getenv("NAVI_CAMERA_INDEX", 0))
CAMERA_RESOLUTION = (640, 480)
TARGET_FPS = 30

# Auto-detect CUDA GPU acceleration if available, fallback to CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- MODEL CONFIGURATION ---
# Removed legacy YOLO dependencies.

# --- PERCEPTION MEMORY ---
MEMORY_DECAY_TIME = float(os.getenv("NAVI_MEMORY_DECAY", 2.0))
# --- AUDIO CONFIGURATION ---
AUDIO_RATE = int(os.getenv("NAVI_AUDIO_RATE", 160))
AUDIO_ENABLED = True

# --- SYSTEM CONFIGURATION ---

# --- NAVIGATION CONFIGURATION ---
DANGER_DISTANCE = float(os.getenv("NAVI_DANGER_DIST", 0.7))    # Normalized relative depth (0.0 - 1.0) where 1.0 is closest
WARNING_DISTANCE = float(os.getenv("NAVI_WARNING_DIST", 0.5))
SPEECH_COOLDOWN = float(os.getenv("NAVI_SPEECH_COOLDOWN", 5.0)) # Seconds before repeating obstacle warning

# --- THREAT DETECTION CONFIGURATION ---
TTC_CRITICAL_THRESHOLD = 1.5 # Seconds to collision
TTC_WARNING_THRESHOLD = 3.0 # Seconds to collision
HUMAN_PROX_VERY_CLOSE = 0.85 # Normalized depth
HUMAN_PROX_CLOSE = 0.7       # Normalized depth

import cv2
import numpy as np
from PIL import Image
from src.vision.interfaces import DepthModel
from config import DEVICE

class DepthAnythingProvider(DepthModel):
    """Handles Depth Anything V2 for state-of-the-art metric depth estimation."""
    
    def __init__(self):
        self.device = DEVICE
        self.active = False
        print(f"[DEPTH-PROVIDER] Loading Depth Anything V2 on '{self.device}'...")
        try:
            from transformers import pipeline
            self.pipe = pipeline(
                task="depth-estimation",
                model="depth-anything/Depth-Anything-V2-Small-hf",
                device=self.device
            )
            # Warm up
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            self.estimate_depth(dummy)
            print("[DEPTH-PROVIDER] Depth Anything V2 loaded successfully!")
            self.active = True
        except Exception as e:
            print(f"[ERROR] Failed to load Depth Anything V2: {e}")
            self.active = False

    def estimate_depth(self, frame: np.ndarray) -> np.ndarray:
        if not self.active: 
            return np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
        try:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(img)
            
            # Inference
            result = self.pipe(image)
            depth_map = np.array(result["depth"])
            
            # Normalize map to 0-1 range for TTC calculations (pseudo-metric)
            # Depth Anything V2 outputs metric or high-quality relative depending on checkpoint.
            max_val = np.max(depth_map)
            if max_val > 0:
                normalized_depth = depth_map / max_val
            else:
                normalized_depth = depth_map.astype(np.float32)
                
            return normalized_depth
        except Exception as e:
            print(f"[DEPTH-PROVIDER] Inference Error: {e}")
            return np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)

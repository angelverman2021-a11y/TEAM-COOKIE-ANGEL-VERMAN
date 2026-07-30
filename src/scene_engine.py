import threading
import time
import cv2
import torch
import sys
import os
from PIL import Image

# Add root directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DEVICE, WARNING_DISTANCE

class SceneUnderstandingEngine:
    """
    Dedicated Asynchronous Module for Florence-2 Large.
    Provides High-Level Scene Understanding without blocking real-time navigation.
    """
    def __init__(self, audio_engine):
        self.audio = audio_engine
        self.active = False
        self.running = False
        self.lock = threading.Lock()
        
        # Internal State
        self.scene_short = "Initializing..."
        self.scene_summary = "Initializing scene understanding..."
        self.last_speech_time = 0
        
        # Start loading model
        threading.Thread(target=self._load_model, daemon=True).start()

    def _load_model(self):
        try:
            from transformers import AutoProcessor, AutoModelForCausalLM
            print("[SCENE-ENGINE] Loading Florence-2 Large (This may take a minute)...")
            
            # Temporary monkey-patch for Florence-2 forced_bos_token_id missing attribute error
            import transformers
            transformers.PretrainedConfig.forced_bos_token_id = None

            self.model_id = 'microsoft/Florence-2-large'
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, trust_remote_code=True).eval().to(DEVICE)
            self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
            self.active = True
            
            print("[SCENE-ENGINE] Florence-2 Large loaded successfully!")
        except Exception as e:
            print(f"[SCENE-ENGINE] Error loading model: {e}")
            self.active = False

    def start(self, vision_engine):
        self.vision = vision_engine
        self.running = True
        threading.Thread(target=self._scene_loop, daemon=True).start()

    def _scene_loop(self):
        while self.running:
            if not self.active:
                time.sleep(1.0)
                continue
                
            frame = self.vision.camera_manager.get_frame()
            if frame is None:
                time.sleep(1.0)
                continue
                
            # Throttle to 1 frame every 2 seconds
            time.sleep(2.0)
            
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(rgb_frame)
                
                # We need both a short CAPTION (for the "scene" field) and a DETAILED_CAPTION (for "summary")
                prompts = ["<CAPTION>", "<DETAILED_CAPTION>"]
                
                results = {}
                for p in prompts:
                    inputs = self.processor(text=p, images=image, return_tensors="pt").to(DEVICE)
                    with torch.no_grad():
                        generated_ids = self.model.generate(
                            input_ids=inputs["input_ids"],
                            pixel_values=inputs["pixel_values"],
                            max_new_tokens=1024,
                            early_stopping=False,
                            do_sample=False,
                            num_beams=3,
                        )
                    generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
                    parsed_answer = self.processor.post_process_generation(
                        generated_text, 
                        task=p, 
                        image_size=(image.width, image.height)
                    )
                    results[p] = parsed_answer[p]
                
                with self.lock:
                    self.scene_short = results["<CAPTION>"]
                    self.scene_summary = results["<DETAILED_CAPTION>"]
                    
                now = time.time()
                if now - self.last_speech_time > 30.0:
                    self.last_speech_time = now
                    # Priority 4 (Scene description)
                    self.audio.speak(self.scene_summary, priority=4)
                    
            except Exception as e:
                print(f"[SCENE-ENGINE] Error during inference: {e}")

    def get_structured_payload(self):
        """Builds the requested JSON output by merging Florence, YOLO, and MiDaS data."""
        if not hasattr(self, 'vision') or not hasattr(self.vision, 'navigation'):
            return {
                "scene": self.scene_short,
                "objects": [],
                "obstacles": [],
                "walkable_direction": "unknown",
                "risk_level": "unknown",
                "summary": self.scene_summary
            }
            
        nav = self.vision.navigation
        with self.lock:
            scene_val = self.scene_short
            summary_val = self.scene_summary
            
        with nav.lock:
            obstacles = [obj["label"] for obj in nav.nav_metrics.values() if obj["dist"] > WARNING_DISTANCE]
            all_objects = [obj["label"] for obj in nav.nav_metrics.values()]
            risk = nav.collision_risk
            safe_dir = nav.safe_direction
            
        return {
            "scene": scene_val,
            "objects": list(set(all_objects)),
            "obstacles": list(set(obstacles)),
            "walkable_direction": safe_dir.lower(),
            "risk_level": risk.lower(),
            "summary": summary_val
        }

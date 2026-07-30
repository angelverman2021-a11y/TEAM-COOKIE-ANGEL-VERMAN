import cv2
import torch
import numpy as np
import threading
import time
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DEVICE,
    DEPTH_MODEL,
    DANGER_DISTANCE,
    WARNING_DISTANCE,
    SPEECH_COOLDOWN,
    NAVIGATION_REFRESH_RATE,
    TTC_CRITICAL_THRESHOLD,
    TTC_WARNING_THRESHOLD,
    HUMAN_PROX_VERY_CLOSE,
    HUMAN_PROX_CLOSE
)

class DepthEstimator:
    """Handles the MiDaS depth estimation model."""
    def __init__(self):
        self.device = DEVICE
        print(f"[NAVIGATION] Loading Depth Model '{DEPTH_MODEL}' on '{self.device}'...")
        try:
            self.model = torch.hub.load("intel-isl/MiDaS", DEPTH_MODEL, trust_repo=True)
            self.model.to(self.device)
            self.model.eval()
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            if DEPTH_MODEL == "DPT_Large" or DEPTH_MODEL == "DPT_Hybrid":
                self.transform = midas_transforms.dpt_transform
            else:
                self.transform = midas_transforms.small_transform
                
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            self.get_depth(dummy)
            print("[NAVIGATION] Depth Model successfully warmed up.")
            self.active = True
        except Exception as e:
            print(f"[ERROR] Failed to load depth model: {e}")
            self.active = False
            
        self.global_max_depth = 1.0

    def get_depth(self, frame):
        if not self.active: return None
        try:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_batch = self.transform(img).to(self.device)
            with torch.no_grad():
                prediction = self.model(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1), size=img.shape[:2], mode="bicubic", align_corners=False,
                ).squeeze()
            depth_map = prediction.cpu().numpy()
            
            current_max = np.max(depth_map)
            if current_max > self.global_max_depth:
                self.global_max_depth = current_max
            else:
                # Decay the global max slowly (2% per frame) so it adjusts to new environments
                self.global_max_depth = self.global_max_depth * 0.98 + current_max * 0.02
                
            # Normalize dynamically safely
            return depth_map / (self.global_max_depth + 1e-6)
        except Exception as e:
            print(f"[NAVIGATION] Depth Inference Error: {e}")
            return None


class CollisionPredictor:
    """Estimates Time-To-Collision (TTC) using depth changes across frames."""
    def __init__(self):
        self.history = {} # track_id -> (timestamp, depth)
        
    def update(self, tracked_objects, current_time):
        results = {}
        for obj in tracked_objects:
            tid = obj["track_id"]
            current_depth = obj["dist"]
            
            ttc = float('inf')
            danger_level = "Safe"
            velocity = 0.0
            
            if tid in self.history:
                prev_time, prev_depth = self.history[tid]
                dt = current_time - prev_time
                if dt > 0:
                    velocity = (current_depth - prev_depth) / dt
                    
                    if velocity > 0.05: # Object is approaching (depth getting closer/larger)
                        # TTC pseudo-metric: Assuming depth=1.0 is collision
                        distance_remaining = max(0, 1.0 - current_depth)
                        ttc = distance_remaining / velocity
                        
                        if ttc < TTC_CRITICAL_THRESHOLD:
                            danger_level = "Critical"
                        elif ttc < TTC_WARNING_THRESHOLD:
                            danger_level = "Warning"
                        elif current_depth > WARNING_DISTANCE:
                            danger_level = "Caution"
            
            self.history[tid] = (current_time, current_depth)
            results[tid] = {"ttc": ttc, "velocity": velocity, "danger_level": danger_level}
            
        # Clean up stale tracks
        active_tids = {obj["track_id"] for obj in tracked_objects}
        self.history = {k: v for k, v in self.history.items() if k in active_tids}
        return results


class ThreatDetector:
    """Detects specific threats like Fast Objects and Human Proximity."""
    def __init__(self, audio_engine):
        self.audio = audio_engine
        self.obstacle_memory = {}
        
    def check_threats(self, tracked_objects, collision_data):
        now = time.time()
        for obj in tracked_objects:
            tid = obj["track_id"]
            label = obj["label"].lower()
            dist = obj["dist"]
            cdata = collision_data.get(tid, {})
            
            # 1. Collision Priority
            if cdata.get("danger_level") == "Critical":
                self._warn(f"collision_{tid}", f"Warning! Possible collision with {label}.", 1, now)
                continue
                
            # 2. Fast Object Priority
            if cdata.get("velocity", 0) > 0.3:
                self._warn(f"fast_{tid}", f"Warning! Fast {label} approaching.", 2, now)
                continue
                
            # 3. Human Proximity
            if label == "person":
                if dist > HUMAN_PROX_VERY_CLOSE:
                    self._warn(f"human_crit_{tid}", "Warning. Person extremely close.", 1, now)
                elif dist > HUMAN_PROX_CLOSE:
                    self._warn(f"human_warn_{tid}", "Person very close.", 5, now)
                elif dist > WARNING_DISTANCE:
                    self._warn(f"human_caut_{tid}", "Person nearby.", 5, now)
                    
    def _warn(self, key, msg, priority, now):
        if key not in self.obstacle_memory or (now - self.obstacle_memory[key]) > SPEECH_COOLDOWN:
            self.obstacle_memory[key] = now
            print(f"[THREAT AUDIO] (Pri {priority}) {msg}")
            self.audio.speak(msg, priority=priority)


class NavigationEngine: # Formerly NavigationManager
    """Orchestrates depth estimation, collision prediction, and threat detection."""
    def __init__(self, audio_engine, vision_engine):
        self.audio = audio_engine
        self.vision = vision_engine
        self.depth_estimator = DepthEstimator()
        self.collision_predictor = CollisionPredictor()
        self.threat_detector = ThreatDetector(audio_engine)
        
        self.running = False
        self.lock = threading.Lock()
        
        # State variables for APIs and HUD
        self.navigation_status = "Initializing"
        self.safe_direction = "Unknown"
        self.obstacle_count = 0
        self.nearest_obstacle = "None"
        self.estimated_distance = 0.0
        self.relative_speed = 0.0
        self.collision_risk = "Low"
        self.danger_level = "Safe"
        self.recommended_action = "Path clear"
        
        self.nav_metrics = {}
        self.obstacle_memory = {}

    def process_frame(self, frame, tracked_objects, now=None):
        if not self.depth_estimator.active:
            self.navigation_status = "Depth Model Failed"
            return
            
        if now is None:
            now = time.time()
            
        depth_map = self.engine_get_depth(frame)
        if depth_map is None:
            self.navigation_status = "Degraded (No Depth)"
            return
            
        h, w = depth_map.shape
        third = w // 3
        
        left_danger = np.mean(np.sort(depth_map[:, :third].flatten())[-int((h*third)*0.1):])
        center_danger = np.mean(np.sort(depth_map[:, third:2*third].flatten())[-int((h*third)*0.1):])
        right_danger = np.mean(np.sort(depth_map[:, 2*third:].flatten())[-int((h*third)*0.1):])
        
        zones = {"Left": left_danger, "Center": center_danger, "Right": right_danger}
        best_dir = min(zones, key=zones.get)
        
        annotated_nav = []
        nearest_dist = 0.0
        nearest_label = "None"
        
        for obj in tracked_objects:
            x1, y1, x2, y2 = obj["box"]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w-1, x2), min(h-1, y2)
            if x2 <= x1 or y2 <= y1: continue
                
            obj_depth_crop = depth_map[y1:y2, x1:x2]
            if obj_depth_crop.size == 0: continue
                
            obj_dist = np.percentile(obj_depth_crop, 90)
            cx = (x1 + x2) / 2
            pos = "Left" if cx < third else "Center" if cx < 2*third else "Right"
            
            nav_obj = {
                "track_id": obj["track_id"],
                "label": obj["label"],
                "dist": obj_dist,
                "pos": pos,
                "box": obj["box"]
            }
            annotated_nav.append(nav_obj)
            
            if obj_dist > nearest_dist:
                nearest_dist = obj_dist
                nearest_label = obj["label"]
        
        # Predict Collisions
        collision_data = self.collision_predictor.update(annotated_nav, now)
        
        # Check Threats
        self.threat_detector.check_threats(annotated_nav, collision_data)
        
        # Aggregate State
        max_danger = "Safe"
        max_vel = 0.0
        risk = "Low"
        action = "Path clear"
        
        for tid, cdata in collision_data.items():
            if cdata["danger_level"] == "Critical": max_danger = "Critical"
            elif cdata["danger_level"] == "Warning" and max_danger != "Critical": max_danger = "Warning"
            elif cdata["danger_level"] == "Caution" and max_danger not in ["Critical", "Warning"]: max_danger = "Caution"
            if abs(cdata["velocity"]) > abs(max_vel): max_vel = cdata["velocity"]
            
        if max_danger == "Critical":
            risk = "High"
            action = "Stop immediately"
            self._nav_advise(action, priority=1, now=now)
        elif max_danger == "Warning":
            risk = "Medium"
            action = f"Move {best_dir.lower()}"
            self._nav_advise(action, priority=3, now=now)
        elif center_danger > DANGER_DISTANCE:
            action = f"Move {best_dir.lower()}"
            self._nav_advise(action, priority=3, now=now)
            
        with self.lock:
            self.safe_direction = best_dir
            self.nearest_obstacle = nearest_label
            self.estimated_distance = nearest_dist
            self.obstacle_count = sum(1 for o in annotated_nav if o["dist"] > WARNING_DISTANCE)
            self.danger_level = max_danger
            self.relative_speed = max_vel
            self.collision_risk = risk
            self.recommended_action = action
            self.nav_metrics = {nav["track_id"]: nav for nav in annotated_nav}
            self.navigation_status = "Active"

    def engine_get_depth(self, frame):
        return self.depth_estimator.get_depth(frame)

    def _nav_advise(self, action, priority, now):
        key = "nav_advice"
        if key not in self.obstacle_memory or (now - self.obstacle_memory[key]) > SPEECH_COOLDOWN:
            self.obstacle_memory[key] = now
            print(f"[NAVI AUDIO] (Pri {priority}) {action}")
            self.audio.speak(action, priority=priority)

    def get_status(self):
        with self.lock:
            return {
                "navigation_status": self.navigation_status,
                "safe_direction": self.safe_direction,
                "obstacle_count": self.obstacle_count,
                "nearest_obstacle": self.nearest_obstacle,
                "estimated_distance": self.estimated_distance,
                "relative_speed": self.relative_speed,
                "collision_risk": self.collision_risk,
                "danger_level": self.danger_level,
                "recommended_action": self.recommended_action,
                "nav_metrics": self.nav_metrics.copy()
            }

    def stop(self):
        self.running = False

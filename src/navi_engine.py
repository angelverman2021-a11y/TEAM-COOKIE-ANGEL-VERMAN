import cv2
import numpy as np
import threading
import collections
import warnings
import time
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.navigation_engine import NavigationEngine
from src.vision.florence_provider import FlorenceProvider
from src.vision.depth_anything_provider import DepthAnythingProvider
from src.perception.memory import TemporalMemory
from src.vision.interfaces import PerceptionResult
from config import (
    CAMERA_INDEX,
    CAMERA_RESOLUTION,
    DEVICE,
    TARGET_FPS,
    MEMORY_DECAY_TIME
)

warnings.filterwarnings("ignore")

class CameraManager:
    """Handles reliable webcam initialization, frame acquisition, and error recovery in a dedicated thread."""
    def __init__(self):
        self.camera = None
        self.running = False
        self.current_frame = None
        self.current_frame_id = 0
        self.frame_lock = threading.Lock()
        
        # Diagnostics
        self.fps_buffer = collections.deque(maxlen=30)
        self.last_time = time.time()
        self.active_index = -1
        self.active_backend = "None"
        self.active_resolution = (0, 0)
        self.status = "Stopped"

    def _update_fps(self):
        now = time.time()
        dt = now - self.last_time
        self.last_time = now
        if dt > 0:
            self.fps_buffer.append(1.0 / dt)
        return sum(self.fps_buffer) / len(self.fps_buffer) if len(self.fps_buffer) > 0 else 0.0

    def start(self):
        self.running = True
        self.status = "Initializing"
        threading.Thread(target=self._camera_loop, daemon=True).start()

    def _initialize_camera(self):
        """Automatically test indices and backends to find a working camera."""
        indices_to_try = [CAMERA_INDEX, 1, 2, 0, 3, -1]
        backends_to_try = [cv2.CAP_DSHOW, cv2.CAP_MSMF, None]
        
        for idx in indices_to_try:
            for backend in backends_to_try:
                if backend is not None:
                    cam = cv2.VideoCapture(idx, backend)
                    backend_name = "DSHOW" if backend == cv2.CAP_DSHOW else "MSMF"
                else:
                    cam = cv2.VideoCapture(idx)
                    backend_name = "DEFAULT"
                
                if cam.isOpened():
                    cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION[0])
                    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION[1])
                    cam.set(cv2.CAP_PROP_FPS, TARGET_FPS)
                    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                    # Test capture
                    for _ in range(5):
                        success, frame = cam.read()
                        if success and frame is not None:
                            self.active_index = idx
                            self.active_backend = backend_name
                            self.active_resolution = (int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                            self.status = "Active"
                            print(f"[CAMERA] Connected idx:{idx} backend:{backend_name} res:{self.active_resolution}")
                            return cam
                        time.sleep(0.05)
                cam.release()
        self.status = "Failed"
        return None

    def _camera_loop(self):
        self.camera = self._initialize_camera()
        
        while self.running:
            if self.camera is None or not self.camera.isOpened():
                self.status = "Reconnecting..."
                time.sleep(1.0)
                self.camera = self._initialize_camera()
                continue

            success, frame = self.camera.read()
            if success and frame is not None:
                with self.frame_lock:
                    self.current_frame = frame
                    self.current_frame_id += 1
                self._update_fps()
            else:
                self.status = "Frame Read Failed"
                self.camera.release()
                self.camera = None

    def get_frame(self):
        with self.frame_lock:
            return (self.current_frame_id, self.current_frame) if self.current_frame is not None else (-1, None)

    def stop(self):
        self.running = False
        if self.camera:
            self.camera.release()

class FrameProcessor:
    """Handles HUD overlays, bounding box drawing."""
    def draw_hud(self, frame, perception: PerceptionResult, nav_status=None):
        annotated = frame.copy()

        # Draw Tracked Bounding Boxes safely
        for obj in perception.objects:
            if "box" not in obj or "label" not in obj:
                continue
            x1, y1, x2, y2 = obj["box"]
            track_id = obj.get("track_id", 0)
            label = obj["label"].upper()
            
            # Default formatting
            color = (0, 255, 100) # Green (Safe)
            dist_str = ""
            pos_str = ""
            
            if nav_status and "nav_metrics" in nav_status and track_id in nav_status["nav_metrics"]:
                nav_data = nav_status["nav_metrics"][track_id]
                dist = nav_data["dist"]
                pos = nav_data["pos"].upper()
                dist_str = f" | DIST:{dist:.2f}"
                pos_str = f" | {pos}"
                
                if dist > 0.8:
                    color = (0, 0, 255) # Red (Critical/Very Close)
                elif dist > 0.6:
                    color = (0, 255, 255) # Yellow (Warning)
                elif dist > 0.4:
                    color = (255, 255, 0) # Cyan (Caution)

            # Draw Box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw Tag
            tag = f"{label}{dist_str}{pos_str}"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, max(0, y1 - 20)), (x1 + tw + 6, max(th + 4, y1)), color, -1)
            text_color = (0, 0, 0) if color != (0, 0, 255) else (255, 255, 255)
            cv2.putText(annotated, tag, (x1 + 3, max(th + 2, y1 - 4)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

        return annotated


class VisionEngine:
    """Main Orchestrator providing live high-FPS streaming decoupled from AI processing."""
    def __init__(self, audio_engine):
        self.audio = audio_engine
        self.running = False
        self.lock = threading.Lock()
        self.diagnostics = {}
        
        self.frame_ready_event = threading.Event()
        self.latest_jpeg_bytes = None
        self.latest_jpeg_id = 0
        self.camera_manager = CameraManager()
        self.processor = FrameProcessor()
        
        # AI State (Thread-Safe)
        self.ai_lock = threading.Lock()
        self.latest_perception = PerceptionResult()
        
        # Diagnostics
        self.ai_time_ms = 0.0

        # Initialize Modular AI Stack
        self.vision_provider = FlorenceProvider()
        self.depth_provider = DepthAnythingProvider()
        self.memory = TemporalMemory(memory_decay_time=MEMORY_DECAY_TIME)
        self.navigation = NavigationEngine(self.audio)
        
    def _ai_loop(self):
        """Dedicated thread for Vision & Depth Provider processing."""
        last_processed_id = -1
        while self.running:
            frame_id, frame = self.camera_manager.get_frame()
            if frame is None or frame_id == last_processed_id:
                time.sleep(0.01)
                continue
            
            last_processed_id = frame_id
            t0 = time.time()
            try:
                # 1. Modular Vision Detection
                perception = self.vision_provider.analyze(frame)
                
                # 2. Modular Depth Estimation
                depth_map = self.depth_provider.estimate_depth(frame)
                perception.depth_map = depth_map
                
                # 3. Update Temporal Memory (Tracks and smoothes)
                stabilized_perception = self.memory.update(perception)
                
                # 4. Depth, Navigation, & Threat Detection (Synchronous)
                self.navigation.process_perception(stabilized_perception)
                
            except Exception as e:
                print(f"[VISION] AI Loop Error: {e}")
                stabilized_perception = PerceptionResult()
            
            with self.ai_lock:
                self.latest_perception = stabilized_perception
                self.ai_time_ms = (time.time() - t0) * 1000

    def _render_loop(self):
        """Dedicated thread to guarantee smooth 30FPS streaming regardless of AI lag."""
        while self.running:
            frame_id, frame = self.camera_manager.get_frame()
            if frame is None:
                # Generate a fallback frame
                frame = np.zeros((CAMERA_RESOLUTION[1], CAMERA_RESOLUTION[0], 3), dtype=np.uint8)
                frame[:] = (0, 0, 150)
                cv2.putText(frame, "WAITING FOR CAMERA...", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            else:
                with self.ai_lock:
                    perception = self.latest_perception
                
                nav_status = self.navigation.get_status()
                
                frame = self.processor.draw_hud(frame, perception, nav_status)
                
                # Save diagnostics for API
                self.diagnostics = {
                    "cam_fps": self.camera_manager._update_fps() if len(self.camera_manager.fps_buffer) > 0 else 0,
                    "cam_status": self.camera_manager.status,
                    "backend": self.camera_manager.active_backend,
                    "resolution": self.camera_manager.active_resolution,
                    "ai_time": self.ai_time_ms,
                    "det_count": len(perception.objects),
                    "device": DEVICE.upper()
                }

            ret, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if ret:
                with self.lock:
                    self.latest_jpeg_bytes = buf.tobytes()
                    self.latest_jpeg_id += 1
                self.frame_ready_event.set()
            
            # Lock render loop to 30 FPS max to save CPU
            time.sleep(1.0 / TARGET_FPS)

    def start(self):
        self.running = True
        self.camera_manager.start()
        
        threading.Thread(target=self._ai_loop, daemon=True).start()
        threading.Thread(target=self._render_loop, daemon=True).start()
        
        self.audio.speak("Modular AI system online. Streaming to dashboard.")

    def get_frame(self):
        with self.lock:
            return (self.latest_jpeg_id, self.latest_jpeg_bytes) if self.latest_jpeg_bytes else (-1, None)

    def stop(self):
        self.running = False
        self.camera_manager.stop()

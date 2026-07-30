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
from src.scene_engine import SceneUnderstandingEngine
from config import (
    CAMERA_INDEX,
    CAMERA_RESOLUTION,
    YOLO_MODEL_PATH,
    CONFIDENCE_THRESHOLD,
    IOU_THRESHOLD,
    DEVICE,
    TARGET_FPS,
    SMOOTHING_BUFFER_SIZE,
    DEBUG_MODE,
    AI_INPUT_SIZE
)

warnings.filterwarnings("ignore")

class CameraManager:
    """Handles reliable webcam initialization, frame acquisition, and error recovery in a dedicated thread."""
    def __init__(self):
        self.camera = None
        self.running = False
        self.current_frame = None
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
                    self.current_frame = frame.copy()
                self._update_fps()
            else:
                self.status = "Frame Read Failed"
                self.camera.release()
                self.camera = None

    def get_frame(self):
        with self.frame_lock:
            return self.current_frame.copy() if self.current_frame is not None else None

    def stop(self):
        self.running = False
        if self.camera:
            self.camera.release()


class Detector:
    """Manages YOLO11 model loading and inference with CUDA/CPU support."""
    def __init__(self, model_path=YOLO_MODEL_PATH, device=DEVICE, conf_thresh=CONFIDENCE_THRESHOLD, iou_thresh=IOU_THRESHOLD, img_size=AI_INPUT_SIZE):
        self.device = device
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.img_size = img_size
        print(f"[VISION-DETECTOR] Loading model '{model_path}' on device '{self.device}'...")
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        
        try:
            self.model.to(self.device)
            # Warm up CUDA
            dummy = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            self.model(dummy, conf=self.conf_thresh, verbose=False)
        except Exception as e:
            print(f"[VISION-DETECTOR] Device warning: {e}. Falling back to CPU.")
            self.device = "cpu"
            self.model.to("cpu")

    def detect(self, frame):
        """Runs object detection with confidence & NMS IOU thresholding."""
        try:
            results = self.model(
                frame,
                conf=self.conf_thresh,
                iou=self.iou_thresh,
                imgsz=self.img_size,
                device=self.device,
                verbose=False
            )
            return results[0] if results else None
        except Exception as e:
            print(f"[VISION-DETECTOR] Inference Error: {e}")
            return None


class Tracker:
    """Persistent object tracking with EMA smoothing to prevent bounding box flicker."""
    def __init__(self, max_lost_frames=10):
        self.next_id = 1
        self.tracked_objects = {} 
        self.max_lost_frames = max_lost_frames
        self.alpha = 0.5  # EMA smoothing factor

    def _compute_iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        if x2 < x1 or y2 < y1: return 0.0
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return intersection / float(area1 + area2 - intersection)

    def _smooth_box(self, old_box, new_box):
        return [
            int(old_box[0] * (1 - self.alpha) + new_box[0] * self.alpha),
            int(old_box[1] * (1 - self.alpha) + new_box[1] * self.alpha),
            int(old_box[2] * (1 - self.alpha) + new_box[2] * self.alpha),
            int(old_box[3] * (1 - self.alpha) + new_box[3] * self.alpha)
        ]

    def update(self, current_detections):
        new_tracked = {}
        for det in current_detections:
            best_id, best_iou = None, 0.3
            for tid, tobj in self.tracked_objects.items():
                if tobj["cls"] != det["cls"]: continue
                iou = self._compute_iou(det["box"], tobj["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_id = tid
            
            if best_id is not None:
                det["box"] = self._smooth_box(self.tracked_objects[best_id]["box"], det["box"])
                new_tracked[best_id] = det
                new_tracked[best_id]["missed"] = 0
                del self.tracked_objects[best_id]
            else:
                new_tracked[self.next_id] = det
                new_tracked[self.next_id]["missed"] = 0
                self.next_id += 1
                
        for tid, tobj in self.tracked_objects.items():
            if tobj["missed"] < self.max_lost_frames:
                tobj["missed"] += 1
                new_tracked[tid] = tobj

        self.tracked_objects = new_tracked
        
        out = []
        for tid, obj in self.tracked_objects.items():
            out.append({"box": obj["box"], "conf": obj["conf"], "cls": obj["cls"], "track_id": tid, "label": obj["label"]})
        return out


class DetectionManager:
    """Orchestrates AI detection, tracking, and metric logging."""
    def __init__(self):
        self.detector = Detector()
        self.tracker = Tracker(max_lost_frames=10)
        self.total_detections = 0
        self.lost_tracks = 0
        self.avg_conf = 0.0

    def process_frame(self, frame):
        result = self.detector.detect(frame)
        current_detections = []
        
        if result and result.boxes is not None:
            for box in result.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                label = result.names[cls_id] if hasattr(result, 'names') else str(cls_id)
                current_detections.append({
                    "box": [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])], 
                    "conf": conf, "cls": cls_id, "label": label
                })
        
        tracked = self.tracker.update(current_detections)
        
        # Update metrics
        self.total_detections += len(current_detections)
        if len(current_detections) > 0:
            avg = sum(d["conf"] for d in current_detections) / len(current_detections)
            self.avg_conf = (self.avg_conf * 0.9) + (avg * 0.1)
            
        self.lost_tracks = sum(1 for obj in self.tracker.tracked_objects.values() if obj["missed"] > 0)
        return tracked


class FrameProcessor:
    """Handles HUD overlays, bounding box drawing."""
    def draw_hud(self, frame, tracked_objects, current_emotion, nav_status=None):
        annotated = frame.copy()

        # Draw Tracked Bounding Boxes safely
        for obj in tracked_objects:
            x1, y1, x2, y2 = obj["box"]
            track_id = obj["track_id"]
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
                
                # Check danger level mapping if collision data exists (we fetch from nav_status.danger_level for now, but that is global)
                # Wait, danger_level is global in nav_status. The individual object doesn't have danger_level returned via api yet.
                # I'll just map color by distance thresholds here since individual TTC isn't available per object in HUD yet.
                if dist > 0.8:
                    color = (0, 0, 255) # Red (Critical/Very Close)
                elif dist > 0.6:
                    color = (0, 255, 255) # Yellow (Warning)
                elif dist > 0.4:
                    color = (255, 255, 0) # Cyan (Caution - Note BGR format: 255 B, 255 G, 0 R -> Cyan)

            # Draw Box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw Tag (No ID, no duplicates)
            tag = f"{label}{dist_str}{pos_str}"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, max(0, y1 - 20)), (x1 + tw + 6, max(th + 4, y1)), color, -1)
            # Text color logic (black on bright colors, white on dark colors)
            text_color = (0, 0, 0) if color != (0, 0, 255) else (255, 255, 255)
            cv2.putText(annotated, tag, (x1 + 3, max(th + 2, y1 - 4)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

        # Removed the permanent Top Banner and Debug HUD logic for a clean feed.
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
        self.camera_manager = CameraManager()
        self.processor = FrameProcessor()
        
        # AI State (Thread-Safe)
        self.ai_lock = threading.Lock()
        self.current_emotion = "Scanning..."
        self.previous_emotion = None
        self.emotion_history = collections.deque(maxlen=SMOOTHING_BUFFER_SIZE)
        self.latest_tracked = []
        self.is_analyzing_emotion = False
        
        # Diagnostics
        self.ai_time_ms = 0.0

        # Initialize AI Models (Takes time, so do it here)
        self.ai_manager = DetectionManager()
        self.navigation = NavigationEngine(self.audio, self)
        self.scene_understanding = SceneUnderstandingEngine(self.audio)
        
        print("[VISION] Warming up DeepFace Engine...")
        try:
            from deepface import DeepFace
            dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
            DeepFace.analyze(dummy_img, actions=['emotion'], enforce_detection=False, detector_backend='skip')
        except Exception as e:
            print(f"[VISION] DeepFace warmup notice: {e}")

    def _analyze_emotion_async(self, person_crop, emotion_callback):
        from deepface import DeepFace
        try:
            analysis = DeepFace.analyze(
                person_crop,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='skip'
            )
            emotion = analysis[0]['dominant_emotion'] if isinstance(analysis, list) else analysis['dominant_emotion']
                
            self.emotion_history.append(emotion)
            with self.ai_lock:
                self.current_emotion = max(set(self.emotion_history), key=self.emotion_history.count)
                
            if self.current_emotion != self.previous_emotion:
                if emotion_callback:
                    emotion_callback(self.current_emotion)
                self.previous_emotion = self.current_emotion
        except Exception:
            pass
        finally:
            self.is_analyzing_emotion = False

    def _ai_loop(self, emotion_callback):
        """Dedicated thread for YOLO and DeepFace processing. Will skip frames if slow."""
        last_processed_frame = None
        while self.running:
            frame = self.camera_manager.get_frame()
            if frame is None or frame is last_processed_frame:
                time.sleep(0.01)
                continue
            
            last_processed_frame = frame
            
            t0 = time.time()
            try:
                # 1. YOLO Detection & Tracking
                tracked = self.ai_manager.process_frame(frame)
                
                # 2. Depth, Navigation, & Threat Detection (Synchronous)
                self.navigation.process_frame(frame, tracked, now=t0)
                
            except Exception as e:
                print(f"[VISION] AI Loop Error: {e}")
                tracked = []
            
            with self.ai_lock:
                self.latest_tracked = tracked
                self.ai_time_ms = (time.time() - t0) * 1000

            person_objs = [obj for obj in tracked if obj["cls"] == 0]
            if person_objs and not self.is_analyzing_emotion:
                best_person = max(person_objs, key=lambda o: (o["box"][2] - o["box"][0]) * (o["box"][3] - o["box"][1]))
                x1, y1, x2, y2 = best_person["box"]
                h, w, _ = frame.shape
                head_y2 = y1 + int(max(1, y2 - y1) * 0.40)
                face_crop = frame[max(0, y1):min(h, head_y2), max(0, x1):min(w, x2)]
                
                if face_crop.size > 0:
                    self.is_analyzing_emotion = True
                    threading.Thread(target=self._analyze_emotion_async, args=(face_crop.copy(), emotion_callback), daemon=True).start()
            elif not person_objs:
                with self.ai_lock:
                    self.current_emotion = "No person detected"

    def _render_loop(self):
        """Dedicated thread to guarantee smooth 30FPS streaming regardless of AI lag."""
        while self.running:
            frame = self.camera_manager.get_frame()
            if frame is None:
                # Generate a fallback frame
                frame = np.zeros((CAMERA_RESOLUTION[1], CAMERA_RESOLUTION[0], 3), dtype=np.uint8)
                frame[:] = (0, 0, 150)
                cv2.putText(frame, "WAITING FOR CAMERA...", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            else:
                with self.ai_lock:
                    tracked = list(self.latest_tracked)
                    emotion = self.current_emotion
                
                nav_status = self.navigation.get_status()
                frame = self.processor.draw_hud(frame, tracked, emotion, nav_status)
                
                # Save diagnostics for API
                self.diagnostics = {
                    "cam_fps": self.camera_manager._update_fps() if len(self.camera_manager.fps_buffer) > 0 else 0,
                    "cam_status": self.camera_manager.status,
                    "backend": self.camera_manager.active_backend,
                    "resolution": self.camera_manager.active_resolution,
                    "ai_time": self.ai_time_ms,
                    "det_count": len(tracked),
                    "device": DEVICE.upper()
                }

            ret, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if ret:
                with self.lock:
                    self.latest_jpeg_bytes = buf.tobytes()
                self.frame_ready_event.set()
            
            # Lock render loop to 30 FPS max to save CPU
            time.sleep(1.0 / TARGET_FPS)

    def start(self, emotion_callback):
        self.running = True
        self.camera_manager.start()
        self.scene_understanding.start(self)
        
        threading.Thread(target=self._ai_loop, args=(emotion_callback,), daemon=True).start()
        threading.Thread(target=self._render_loop, daemon=True).start()
        
        self.audio.speak("Vision system online. Streaming to dashboard.")

    def get_frame(self):
        with self.lock:
            return self.latest_jpeg_bytes

    def stop(self):
        self.running = False
        self.camera_manager.stop()
        self.navigation.stop()

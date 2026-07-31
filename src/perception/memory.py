import time
from src.vision.interfaces import PerceptionResult
from typing import List
class TemporalMemory:
    """
    Lightweight software perception memory.
    Persists detections and scene state across frames.
    """
    def __init__(self, memory_decay_time: float = 2.0, alpha: float = 0.5):
        self.memory_decay_time = memory_decay_time
        self.alpha = alpha  # EMA smoothing factor for bounding boxes
        self.next_id = 1
        
        # State
        self.tracked_objects = {}  # tid -> dict with 'last_seen', 'data'
        self.current_scene_status = "Scanning..."
        self.current_ocr = ""
        self.current_nav_context = ""
        self.last_scene_update = 0.0
        
    def _compute_iou(self, box1: List[int], box2: List[int]) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        if x2 < x1 or y2 < y1: return 0.0
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return intersection / float(area1 + area2 - intersection + 1e-6)

    def _smooth_box(self, old_box: List[int], new_box: List[int]) -> List[int]:
        return [
            int(old_box[0] * (1 - self.alpha) + new_box[0] * self.alpha),
            int(old_box[1] * (1 - self.alpha) + new_box[1] * self.alpha),
            int(old_box[2] * (1 - self.alpha) + new_box[2] * self.alpha),
            int(old_box[3] * (1 - self.alpha) + new_box[3] * self.alpha)
        ]

    def update(self, perception: PerceptionResult) -> PerceptionResult:
        """
        Takes a new PerceptionResult from the VisionModel and DepthModel, updates memory,
        and returns a stabilized PerceptionResult.
        """
        now = time.time()
        
        # 1. Update global scene state if fresh
        if perception.scene_status != "Scanning..." and perception.scene_status != "":
            self.current_scene_status = perception.scene_status
            self.current_ocr = perception.ocr_text
            self.current_nav_context = perception.navigation_context
            self.last_scene_update = now

        # 2. Update object memory
        new_tracked = {}
        for det in perception.objects:
            best_id, best_iou = None, 0.3
            for tid, tobj in self.tracked_objects.items():
                if tobj["data"]["label"] != det["label"]: continue
                iou = self._compute_iou(det["box"], tobj["data"]["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_id = tid
            
            if best_id is not None:
                det["box"] = self._smooth_box(self.tracked_objects[best_id]["data"]["box"], det["box"])
                det["track_id"] = best_id
                new_tracked[best_id] = {"last_seen": now, "data": det}
                del self.tracked_objects[best_id]
            else:
                det["track_id"] = self.next_id
                new_tracked[self.next_id] = {"last_seen": now, "data": det}
                self.next_id += 1
                
        # Keep old objects if they haven't expired
        for tid, tobj in self.tracked_objects.items():
            if (now - tobj["last_seen"]) < self.memory_decay_time:
                new_tracked[tid] = tobj

        self.tracked_objects = new_tracked
        
        # Build stabilized result
        stabilized_objects = [obj["data"] for obj in self.tracked_objects.values()]
        
        return PerceptionResult(
            objects=stabilized_objects,
            scene_status=self.current_scene_status,
            ocr_text=self.current_ocr,
            navigation_context=self.current_nav_context,
            depth_map=perception.depth_map, # Pass through latest depth map
            timestamp=now
        )

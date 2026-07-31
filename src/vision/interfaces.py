from abc import ABC, abstractmethod
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import time

@dataclass
class PerceptionResult:
    """Standardized object representing the complete perception state."""
    objects: List[Dict[str, Any]] = field(default_factory=list)
    scene_status: str = "Scanning..."
    ocr_text: str = ""
    landmarks: List[str] = field(default_factory=list)
    depth_map: Optional[np.ndarray] = None
    navigation_context: str = ""
    hazards: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

class VisionModel(ABC):
    """Abstract interface for a Vision Foundation Model."""
    
    @abstractmethod
    def analyze(self, frame: np.ndarray) -> PerceptionResult:
        """
        Analyze a video frame and return a standardized PerceptionResult.
        Must handle object detection, OCR, and scene understanding.
        """
        pass

class DepthModel(ABC):
    """Abstract interface for a Monocular Depth Model."""
    
    @abstractmethod
    def estimate_depth(self, frame: np.ndarray) -> np.ndarray:
        """
        Analyze a video frame and return a normalized depth map (0-255).
        """
        pass

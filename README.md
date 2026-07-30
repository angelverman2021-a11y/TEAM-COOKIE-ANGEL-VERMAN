# NAVI: AI Smart Glasses Server

## Overview

NAVI is a real-time, deterministic computer vision server designed to provide spatial awareness, obstacle avoidance, and scene understanding for visually impaired users via smart glasses. The system processes a live video feed, extracting geometric and semantic information to issue priority-queued audio navigation commands.

## System Architecture

The core of NAVI operates on a strictly ordered, synchronous pipeline designed to maximize deterministic outcomes while preserving real-time performance.

The frame processing flow executes in the following sequence:

1. **Camera Input**: High-speed frame acquisition via OpenCV.
2. **Object Detection**: YOLOv11 extracts localized bounding boxes and classification labels.
3. **Object Tracking**: ByteTrack assigns stable, temporal IDs to bounding boxes across frames.
4. **Depth Estimation**: The MiDaS neural network generates a normalized depth map from the exact same frame.
5. **Navigation Decision**: The system calculates the 90th percentile depth of each tracked object to estimate distance. The view is divided into trisection zones (Left, Center, Right) to determine the safest walkable vector.
6. **Threat Detection**: The engine calculates Time-To-Collision (TTC) using depth velocity and flags immediate hazards such as rapidly approaching objects or human proximity violations.
7. **Speech Decision Engine**: Actionable intelligence is mapped to discrete warning levels (Critical, Warning, Caution, Safe).
8. **Speech Priority Queue**: Audio messages are dispatched into an asynchronous priority queue.
9. **Audio Output**: SAPI5 (via pyttsx3) synthesizes the speech. High-priority interrupts (e.g., imminent collision) can preempt low-priority speech mid-sentence.

*Note: Scene Understanding (Florence-2) and Emotion Recognition (DeepFace) run asynchronously to prevent pipeline blocking.*

## Technology Stack

### Core System
- **Language**: Python 3.11+
- **Hardware Acceleration**: PyTorch (CUDA supported)
- **Web Interface**: Flask, HTML5, Vanilla JavaScript, CSS

### AI & Machine Learning Models
- **Ultralytics YOLOv11**: Real-time object detection.
- **ByteTrack**: High-performance multi-object tracking.
- **MiDaS (intel-isl/MiDaS)**: Monocular depth estimation for spatial mapping.
- **Microsoft Florence-2 Large**: Vision-language model for rich, high-level scene understanding and structured summarization.
- **DeepFace**: Facial recognition and emotion detection.

### Audio & I/O
- **pyttsx3**: Cross-platform Text-to-Speech synthesis with priority threading.
- **OpenCV**: Video capture, frame preprocessing, and stream encoding.

## Project Structure

- `main.py`: The entry point. Initializes the Flask server, exposes REST APIs (`/api/status`), and serves the dashboard.
- `config.py`: Centralized configuration (thresholds, hardware parameters, cooldowns).
- `src/navi_engine.py`: Contains the `VisionEngine` which orchestrates the primary synchronous AI loop and decoupled camera rendering.
- `src/navigation_engine.py`: Houses the logic for spatial calculations, collision prediction, and the `ThreatDetector`.
- `src/scene_engine.py`: Asynchronous runner for the Florence-2 model.
- `src/audio_engine.py`: Thread-safe, interruptible priority queue for TTS operations.
- `templates/` & `static/`: Frontend assets for the live telemetry dashboard.

## Installation and Execution

### Prerequisites
- Python 3.11 or higher.
- NVIDIA GPU with CUDA Toolkit (highly recommended for real-time inference).

### Execution

To launch the backend server and initialize the AI models:

**Windows**:
```bat
./run.bat
```

**Manual Execution**:
```bash
python main.py
```

Upon execution, the server will load all PyTorch models into VRAM and expose the web dashboard on `http://127.0.0.1:5000`. 

## Shared AI State API

The system exposes telemetry via a high-speed polling endpoint at `/api/status`. The payload includes:
- Aggregated Scene Summary (from Florence-2)
- Detected Emotion (from DeepFace)
- Bounding boxes, distances, and tracking IDs (from YOLO/MiDaS)
- Collision risk, recommended actions, and safe direction vectors (from NavigationEngine)
- System health and thread diagnostics

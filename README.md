# NAVI: Navigation Assistant for the Visually Impaired 👁️

**NAVI** is an AI-powered, wearable assistant designed to solve "Social Blindness" for visually impaired individuals. While existing tools identify physical objects (like chairs or doors), NAVI identifies the **social context** of a room. 

It tells the user if the person in front of them is smiling, angry, or waiting for them to speak, allowing them to navigate social situations with confidence.

---

## 🚀 How It Works
NAVI uses a highly optimized, real-time multimodal pipeline:
1. **Vision System (`navi_engine.py`)**: Uses **YOLOv8** for real-time person detection.
2. **Context Engine (DeepFace)**: Uses **RetinaFace** to perfectly crop the face, then extracts micro-expressions using a Convolutional Neural Network.
3. **Audio Output (`audio_engine.py`)**: Synthesizes the contextual data into a human-friendly sentence and speaks it aloud instantly using offline Text-to-Speech (`pyttsx3`).

*Note: The engine uses advanced temporal smoothing and threading to ensure it runs at 30+ FPS locally without any cloud compute.*

---

## 🛠️ Setup Instructions

### 1. Requirements
Ensure you have Python installed, then set up your environment:
```bash
python -m venv venv
venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics opencv-python opencv-contrib-python deepface pyttsx3
```

### 3. Run the App
To start the full NAVI Orchestrator (Vision + Audio):
```bash
python main.py
```

*Press `q` on your keyboard while selecting the video window to quit.*

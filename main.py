from flask import Flask, render_template, Response, jsonify, request
from src.navi_engine import VisionEngine
from src.audio_engine import AudioEngine
import threading
import time

app = Flask(__name__)

# Global State for Web Dashboard
system_status = "Disconnected"
guardian_info = {"name": "", "phone": ""}

# Lazily loaded engines
audio = None
vision = None

def get_audio():
    global audio
    if audio is None:
        audio = AudioEngine()
    return audio

def get_vision():
    global vision
    if vision is None:
        vision = VisionEngine(audio_engine=get_audio())
    return vision

def generate_video_stream():
    """Generator function to yield JPEG frames instantly whenever a new frame is ready."""
    last_frame_id = -1
    v = get_vision()
    while True:
        if v.running:
            v.frame_ready_event.wait(timeout=0.1)
            v.frame_ready_event.clear()
        
        frame_id, frame = v.get_frame()
        if frame is not None and frame_id != last_frame_id:
            last_frame_id = frame_id
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            time.sleep(0.01)

# --- FLASK ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status', methods=['GET'])
def get_status():
    v = get_vision()
    nav_status = v.navigation.get_status() if hasattr(v, 'navigation') else {}
    diagnostics = v.diagnostics if hasattr(v, 'diagnostics') else {}
    
    # Read from unified PerceptionResult
    perception = v.latest_perception if hasattr(v, 'latest_perception') else None
    scene_status = perception.scene_status if perception else "Scanning..."
    ocr_text = perception.ocr_text if perception else ""
    nav_context = perception.navigation_context if perception else ""
    
    return jsonify({
        "scene_status": scene_status,
        "status": system_status,
        "guardian_set": bool(guardian_info["name"]),
        "navigation_status": nav_status.get("navigation_status", "Unknown"),
        "safe_direction": nav_status.get("safe_direction", "Unknown"),
        "obstacle_count": nav_status.get("obstacle_count", 0),
        "nearest_object": nav_status.get("nearest_obstacle", "None"),
        "nearest_distance": nav_status.get("estimated_distance", 0.0),
        "relative_speed": nav_status.get("relative_speed", 0.0),
        "collision_risk": nav_status.get("collision_risk", "Low"),
        "danger_level": nav_status.get("danger_level", "Safe"),
        "recommended_action": nav_status.get("recommended_action", "Path clear"),
        "diagnostics": diagnostics,
        "nav_metrics": nav_status.get("nav_metrics", {}),
        "scene_understanding": {
            "scene": scene_status,
            "ocr": ocr_text,
            "summary": nav_context
        }
    })

@app.route('/api/guardian', methods=['POST'])
def set_guardian():
    data = request.json
    guardian_info["name"] = data.get("name", "")
    guardian_info["phone"] = data.get("phone", "")
    get_audio().speak(f"Guardian {guardian_info['name']} has been linked to your NAVI glasses.")
    return jsonify({"success": True})

@app.route('/api/sos', methods=['POST'])
def trigger_sos():
    if guardian_info["name"]:
        get_audio().speak(f"SOS triggered. Sending location and camera feed to {guardian_info['name']}.")
        return jsonify({"success": True, "message": f"Alert sent to {guardian_info['name']}"})
    else:
        get_audio().speak("SOS triggered, but no guardian is configured.")
        return jsonify({"success": False, "message": "No Guardian Configured"})

@app.route('/api/connect', methods=['POST'])
def connect_glasses():
    global system_status
    v = get_vision()
    if not v.running:
        system_status = "Booting Vision Engine..."
        # Start vision engine in a background thread so it doesn't block Flask
        threading.Thread(target=v.start, daemon=True).start()
        return jsonify({"success": True, "message": "Connected successfully"})
    return jsonify({"success": True, "message": "Already connected"})

if __name__ == "__main__":
    print("=========================================")
    print("    Starting NAVI Web Dashboard...       ")
    print("    Go to http://127.0.0.1:5000          ")
    print("=========================================")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

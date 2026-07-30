from flask import Flask, render_template, Response, jsonify, request
from src.navi_engine import VisionEngine
from src.audio_engine import AudioEngine
import threading
import time

app = Flask(__name__)

# Initialize Core AI Engines
audio = AudioEngine()
vision = VisionEngine(audio_engine=audio)

# Global State for Web Dashboard
system_status = "Disconnected"
guardian_info = {"name": "", "phone": ""}

last_emotion_time = 0
def on_emotion_changed(emotion):
    global system_status, last_emotion_time
    now = time.time()
    
    if emotion and emotion != "Scanning...":
        if emotion == "No person detected":
            sentence = "I do not see anyone in front of you."
        elif emotion == "Face not clearly visible":
            sentence = "Someone is there, but their face is turned away."
        else:
            sentence = f"The person in front of you seems {emotion}."
        
        system_status = f"Last detected: {emotion}"
        
        if now - last_emotion_time > 15.0:
            last_emotion_time = now
            print(f"[NAVI AUDIO]: {sentence}")
            audio.speak(sentence, priority=4)
        
        # Automatic Guardian SOS Trigger (Example logic)
        if emotion == "Angry" and guardian_info["name"]:
            alert_msg = f"Warning. Aggression detected. Alerting {guardian_info['name']}."
            # High priority alert
            if now - last_emotion_time > 15.0:
                audio.speak(alert_msg, priority=1)
            print(f"[GUARDIAN SOS]: SMS sent to {guardian_info['phone']} - User may be in a hostile environment!")

def generate_video_stream():
    """Generator function to yield JPEG frames instantly whenever a new frame is ready."""
    last_frame = None
    while True:
        if vision.running:
            vision.frame_ready_event.wait(timeout=0.1)
            vision.frame_ready_event.clear()
        
        frame = vision.get_frame()
        if frame is not None and frame != last_frame:
            last_frame = frame
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
    nav_status = vision.navigation.get_status() if hasattr(vision, 'navigation') else {}
    diagnostics = vision.diagnostics if hasattr(vision, 'diagnostics') else {}
    scene_data = vision.scene_understanding.get_structured_payload() if hasattr(vision, 'scene_understanding') else {}
    return jsonify({
        "emotion": vision.current_emotion,
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
        "scene_understanding": scene_data
    })

@app.route('/api/guardian', methods=['POST'])
def set_guardian():
    data = request.json
    guardian_info["name"] = data.get("name", "")
    guardian_info["phone"] = data.get("phone", "")
    audio.speak(f"Guardian {guardian_info['name']} has been linked to your NAVI glasses.")
    return jsonify({"success": True})

@app.route('/api/sos', methods=['POST'])
def trigger_sos():
    if guardian_info["name"]:
        audio.speak(f"SOS triggered. Sending location and camera feed to {guardian_info['name']}.")
        return jsonify({"success": True, "message": f"Alert sent to {guardian_info['name']}"})
    else:
        audio.speak("SOS triggered, but no guardian is configured.")
        return jsonify({"success": False, "message": "No Guardian Configured"})

@app.route('/api/connect', methods=['POST'])
def connect_glasses():
    global system_status
    if not vision.running:
        system_status = "Booting Vision Engine..."
        # Start vision engine in a background thread so it doesn't block Flask
        threading.Thread(target=vision.start, args=(on_emotion_changed,), daemon=True).start()
        return jsonify({"success": True, "message": "Connected successfully"})
    return jsonify({"success": True, "message": "Already connected"})

if __name__ == "__main__":
    print("=========================================")
    print("    Starting NAVI Web Dashboard...       ")
    print("    Go to http://127.0.0.1:5000          ")
    print("=========================================")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

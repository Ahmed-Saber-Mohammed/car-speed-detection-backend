from flask import Flask, request, jsonify
import numpy as np
import cv2
from shared_state import SharedState


app = Flask(__name__)


@app.route("/upload_video", methods=["POST"])
def upload_video():
    """Receives video frames and updates the latest frame in memory."""
    file = request.files.get("video")
    if file is None:
        print("[API] No file in request")
        return "No file", 400

    print("[API] Received video frame")
    
    nparr = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        print("[API] Failed to decode image")
        return "Failed to decode frame", 400

    frame = cv2.resize(frame, (640, 480))  # Resize for performance

    with SharedState.frame_lock:
        SharedState.latest_frame = frame.copy()  # Store latest frame

    return "Frame received", 200


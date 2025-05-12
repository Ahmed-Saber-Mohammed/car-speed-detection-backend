from flask import Flask, request, jsonify
import numpy as np
import cv2
from shared_state import SharedState
import config
from supabase import create_client, Client


app = Flask(__name__)

if not all([config.SUPABASE_URL, config.SUPABASE_KEY, config.BUCKET_NAME]): 
    raise ValueError("Supabase credentials not found")


supabase: Client = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)

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

@app.route("/set_speed_limit", methods=["POST"])
def set_speed_limit():
    
    try:
        data = request.get_json()
        print(f"📥 Received API request: {data}")  # Debugging print

        new_limit = int(data["max_speed"])  # Extract speed limit
        if not (10 <= new_limit <= 220):  # Validate range
            print(f"🚨 Rejected invalid speed limit: {new_limit}")
            return {
                "error": "Invalid speed limit range (must be between 10 and 220 km/h)"
            }, 400

        with SharedState.speed_limit_lock:  # Use lock to safely update
            SharedState.speedLimit = new_limit

        print(f"✅ Speed limit manually set to {SharedState.speedLimit} km/h")
        return {"message": "Speed limit updated", "SharedState.speedLimit": SharedState.speedLimit}, 200
    except (KeyError, ValueError):
        print("❌ Invalid request format")
        return {"error": "Invalid speed limit value"}, 400

@app.route("/overspeeding_cars", methods=["GET"])
def get_overspeeding_cars():
    response = supabase.table("overspeeding_cars").select("*").execute()
    
    # Debugging: Print the response
    print(response)

    # Ensure data is correctly returned
    if isinstance(response, tuple):  # Handle tuple response
        data, error = response
        if error:
            return jsonify({"error": str(error)}), 500
        return jsonify(data), 200

    return jsonify(response.data), 200  # Standard APIResponse handling

@app.route("/overspeeding_cars/<int:car_id>", methods=["DELETE"])
def delete_overspeeding_car(car_id):
    """Delete a car entry from the Supabase table and remove its image from storage."""
    try:
        # Fetch car entry from Supabase
        response = supabase.table("overspeeding_cars").select("id", "image_path").execute()

        if not response.data:
            print("⚠️ No cars found in Supabase.")
            return jsonify({"error": "Car not found"}), 404

        print(f"📊 Current Cars in DB: {response.data}")

        car_entry = next((item for item in response.data if item["id"] == car_id), None)
        if not car_entry:
            print(f"⚠️ Car ID {car_id} not found in Supabase.")
            return jsonify({"error": "Car not found"}), 404

        # Get image path from Supabase record
        image_path = car_entry["image_path"]
        print(f"🖼 Image Path: {image_path}")

        # Get file path directly from response.full_path
        response = supabase.storage.from_(config.BUCKET_NAME).list()
        image_file = next(
            (file["name"] for file in response if file["name"] in image_path),
            None
        )

        if not image_file:
            print(f"⚠️ Warning: Image file not found in storage: {image_path}")
            return jsonify({"error": "Image file not found"}), 404

        print(f"🗑 Deleting Image: {image_file}")

        # Delete car record
        delete_response = supabase.table("overspeeding_cars").delete().eq("id", car_id).execute()
        if delete_response.data:
            print(f"✅ Deleted car record: {delete_response.data}")
        else:
            print(f"⚠️ Failed to delete car record: {delete_response}")
            return jsonify({"error": "Failed to delete car record"}), 500

        # Delete image
        storage_response = supabase.storage.from_(config.BUCKET_NAME).remove([image_file])
        print(f"📦 Storage Delete Response: {storage_response}")

        return jsonify({"message": "Car deleted successfully"}), 200

    except Exception as e:
        print(f"❌ Exception: {e}")
        return jsonify({"error": str(e)}), 500

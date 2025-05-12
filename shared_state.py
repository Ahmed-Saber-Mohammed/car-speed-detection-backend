# shared_state.py
from threading import Lock

class SharedState:
    latest_frame = None
    frame_lock = Lock()
    speed_limit_lock = Lock()
    speedLimit = 20 
    image_url = None
    saved_cars = set()  # <- to track already saved overspeeding cars
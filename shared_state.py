# shared_state.py
from threading import Lock

class SharedState:
    latest_frame = None
    frame_lock = Lock()

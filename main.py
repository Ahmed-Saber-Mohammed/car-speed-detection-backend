from threading import Thread
from detection import detect_and_track
from api import app


def run_flask():
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)

if __name__ == "__main__":
    
    Thread(target=detect_and_track, daemon=True).start()
    run_flask()
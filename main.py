from threading import Thread
from detection import *
from api import app


def run_flask():
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)

if __name__ == "__main__":
    Thread(target=update_speed_limit, daemon=True).start()
    Thread(target=detect_and_track, daemon=True).start()
    run_flask()
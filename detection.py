import cv2
from ultralytics import YOLO
import pandas as pd
import time
import config
from tracker import Tracker
from shared_state import SharedState
import requests
from datetime import datetime
from supabase import create_client, Client
import asyncio


if not all([config.SUPABASE_URL, config.SUPABASE_KEY, config.BUCKET_NAME]): 
    raise ValueError("Supabase credentials not found")


supabase: Client = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)


def load_model(model_path):
    return YOLO(model_path)

def load_class_list(class_list_path):
    with open(class_list_path, "r") as file:
        return file.read().split("\n")

def calculate_speed(elapsed_time, distance):
    if elapsed_time > 0:
        speed_ms = distance / elapsed_time
        speed_km = speed_ms * 3.6
        return speed_km
    return 0

def display_speed(frame, cx, cy, vehicle_id, speed_km, x, y):
    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
    if speed_km > 0:
        cv2.putText(frame, str(int(speed_km)) + " Km/h", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
    else:
        cv2.putText(frame, "Speed N/A", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

def update_speed_limit():

    api_url = f"{config.BASE_URL}/setspeedlimit"  # Replace with actual API URL

    while True:
        try:
            response = requests.get(api_url, timeout=5)  # Fetch new speed limit
            if response.status_code == 200:
                data = response.json()
                print(f"🔍 API Response (Fetched Speed): {data}")  # Log API response

                if "max_speed" in data:
                    new_limit = int(data["max_speed"])  # Extract speed limit

                    if not (10 <= new_limit <= 200):  # Ensure valid range
                        print(f"🚨 Ignored invalid speed limit: {new_limit}")
                        continue

                    with SharedState.speed_limit_lock:  # Ensure safe update
                        if new_limit != SharedState.speedLimit:
                            SharedState.speedLimit = new_limit
                            print(f"✅ Updated speed limit to {SharedState.speedLimit} km/h")
        except requests.RequestException as e:
            print(f"⚠️ Failed to fetch speed limit: {e}")

        time.sleep(60)  # Check for updates every 60 seconds

def process_frame(frame, model, class_list, tracker, times, speed_memory):
    cy1 = 322
    cy2 = 368
    offset = 6

    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a.cpu()).astype("float")

    vehicles = []
    for index, row in px.iterrows():
        x1, y1, x2, y2, _, d = map(int, row)
        c = class_list[d]
        if c in ["car", "truck", "bus"]:
            vehicles.append([x1, y1, x2, y2])

    bbox_id = tracker.update(vehicles)

    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        cx = (x3 + x4) // 2
        cy = (y3 + y4) // 2

        if cy1 - offset < cy < cy1 + offset:
            times[id] = time.time()

        if cy2 - offset < cy < cy2 + offset and id in times:
            elapsed_time = time.time() - times[id]
            speed = round(calculate_speed(elapsed_time, 10),1)
            speed_memory[id] = speed
            del times[id]
            # 🚨 Check and Save Overspeeding Cars
            if speed > SharedState.speedLimit:
                # Async call must be awaited using asyncio
                asyncio.run(saveCar(id, speed, frame, x3, y3, x4 - x3, y4 - y3))


        if id in speed_memory:
            display_speed(frame, cx, cy, id, speed_memory[id], x4, y4)

        cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 1)

    cv2.line(frame, (267, cy1), (829, cy1), (255, 255, 255), 1)
    cv2.putText(frame, "1line", (274, 318), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

    cv2.line(frame, (167, cy2), (932, cy2), (255, 255, 255), 1)
    cv2.putText(frame, "2line", (181, 363), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

    return frame

def detect_and_track():
    
    model = load_model(config.MODEL_PATH)
    class_list = load_class_list(config.COCO_CLASSES_PATH)
    # cap = cv2.VideoCapture(0)
    tracker = Tracker()
    speed_memory = {}
    times = {}
    saved_cars = set()     # to prevent saving same car multiple times

    while True:
        frame = None
        
        with SharedState.frame_lock:
            if SharedState.latest_frame is not None:
                frame = SharedState.latest_frame.copy()
          
        if frame is None:
            # print("No frame yet")
            continue

        frame = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
        frame = process_frame(frame, model, class_list, tracker,times,speed_memory)



        print("[DETECT] Got a frame of size:", frame.shape)

        cv2.imshow("RGB", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()

async def saveCar(carID, speed, frame, tx, ty, tw, th):
    """Saves a cropped image of an overspeeding car to Supabase storage."""
    now = datetime.now()
    filename = now.strftime(f"%d-%m-%Y-%H-%M-%S-{speed}")
    image_filename = f"{filename}.jpeg"

    try:
        # Crop the car image from the frame
        car_image = frame[ty : ty + th, tx : tx + tw]

        # Check if the cropped image is valid
        if car_image.size == 0:
            print(f"⚠️ Error: Cropped car image is empty. Skipping upload.")
            return None

        # Draw red box on the *cropped* car image
        cv2.rectangle(car_image, (0, 0), (tw, th), (0, 0, 255), 3)

        # Calculate text position *relative to original frame* and then adjust for crop
        text_x = 5 # tx - tx  # Position the text a little to the left of the car
        text_y = -5 #ty - ty - 10  # Position the text above the car

        # Ensure text position is within cropped image boundaries
        if 0 <= text_x < car_image.shape[1] and 0 <= text_y < car_image.shape[0]:
            # Overlay text on the *cropped* car image
            cv2.putText(
                car_image,
                f"OVERSPEEDING {speed} km/h",
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,  # Adjusted scale for the cropped image
                (0, 0, 255),
                2,
            )
        else:
            print(
                "⚠️ Warning: Text position is outside the cropped image "
                f"boundaries.  Text_x: {text_x}, text_y: {text_y},  Shape: {car_image.shape}"
            )

        # Convert the cropped car image to JPEG bytes
        _, img_encoded = cv2.imencode(".jpeg", car_image)
        image_bytes = img_encoded.tobytes()

        # Upload the image to Supabase storage
        response = supabase.storage.from_(config.BUCKET_NAME).upload(
            image_filename,
            image_bytes,
            file_options={
                "cacheControl": "3600",
                "upsert": False,
                "contentType": "image/jpeg",
            },
        )  # Set contentType
        print(response.__dict__)  # Show all attributes and values


        if hasattr(response, 'full_path') and response.full_path:
          print("Upload successful:", response.full_path)
        BASE_IMAGE_URL=f"{config.SUPABASE_URL}/storage/v1/object/public/"
        SharedState.image_url = f"{BASE_IMAGE_URL}{response.full_path}"

        data = {
                "image_path": SharedState.image_url,
                "speed": speed,
                "date": now.strftime("%d/%m/%Y"),
                "time": now.strftime("%H:%M:%S"),
            }
        insert_response = supabase.table("overspeeding_cars").insert(data).execute()

        if insert_response.data:
                print("✅ Data inserted into Supabase table:", insert_response.data)
        else:
                print("⚠️ Error inserting data into Supabase table:", insert_response)

        return SharedState.image_url  # Return the URL

    except Exception as e:
        print(f"An error occurred during cropping/upload: {e}")
        return None  # Indicate failure

import cv2
import face_recognition
import pickle
import os
import time

# Path to store encodings
ENCODINGS_FILE = "encodings.pickle"
# Directory to save sample images (optional)
IMAGE_SAMPLES_DIR = "image_samples" 

def capture_image_from_camera(camera_index=0, display=True):
    """Captures a single image from the specified camera."""
    print(f"[INFO] Initializing camera {camera_index}...")
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {camera_index}.")
        return None

    print("[INFO] Camera initialized. Press 'c' to capture, 'q' to quit.")
    
    img_captured = None
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame.")
            time.sleep(0.1) # Avoid busy-waiting
            continue

        if display:
            # Display instructions on the frame
            cv2.putText(frame, "Press 'c' to capture, 'q' to quit", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Capture Face - Press 'c'", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            img_captured = frame.copy()
            print("[INFO] Image captured!")
            break
        elif key == ord('q'):
            print("[INFO] Quitting capture.")
            break
            
    cap.release()
    if display:
        cv2.destroyAllWindows()
        # Need a small delay for windows to close properly on some systems
        for i in range(5):
            cv2.waitKey(1)
            
    return img_captured

def save_sample_image(image, name):
    """Saves the captured image to the samples directory (optional)."""
    if not os.path.exists(IMAGE_SAMPLES_DIR):
        os.makedirs(IMAGE_SAMPLES_DIR)
        print(f"[INFO] Created directory: {IMAGE_SAMPLES_DIR}")
        
    # Create a unique filename
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(IMAGE_SAMPLES_DIR, f"{name}_{timestamp}.png")
    
    try:
        cv2.imwrite(filename, image)
        print(f"[INFO] Sample image saved to {filename}")
    except Exception as e:
        print(f"[ERROR] Could not save sample image: {e}")


def register_new_face(name, image):
    """Detects face, extracts encoding, and saves it."""
    print(f"[INFO] Processing image for {name}...")
    
    # Convert image from BGR (OpenCV default) to RGB (face_recognition default)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect face locations. Use 'hog' model for CPU (faster on Pi), 'cnn' for GPU/TPU
    # Consider increasing model accuracy if needed: face_locations(rgb_image, number_of_times_to_upsample=2, model='hog')
    boxes = face_recognition.face_locations(rgb_image, model='hog') 

    if not boxes:
        print("[ERROR] No face detected in the image. Please try again.")
        return False
    if len(boxes) > 1:
        print("[ERROR] Multiple faces detected. Please ensure only one face is clearly visible.")
        return False

    # Extract face encodings
    encodings = face_recognition.face_encodings(rgb_image, boxes)

    if not encodings:
         print("[ERROR] Could not generate encoding for the detected face.")
         return False
         
    print("[INFO] Face detected and encoding generated.")
    
    # Load existing encodings or initialize new list
    if os.path.exists(ENCODINGS_FILE):
        print(f"[INFO] Loading existing encodings from {ENCODINGS_FILE}...")
        try:
            with open(ENCODINGS_FILE, "rb") as f:
                data = pickle.load(f)
        except (pickle.UnpicklingError, EOFError) as e:
             print(f"[WARNING] Could not load existing encodings file ({e}). Starting fresh.")
             data = {"encodings": [], "names": []}
        except Exception as e:
             print(f"[ERROR] An unexpected error occurred loading encodings: {e}")
             print("[INFO] Starting with fresh encodings.")
             data = {"encodings": [], "names": []}
    else:
        print("[INFO] No existing encodings found. Creating new file.")
        data = {"encodings": [], "names": []}

    # Check if name already exists
    if name in data["names"]:
        overwrite = input(f"[WARNING] Name '{name}' already exists. Overwrite? (y/n): ").lower()
        if overwrite == 'y':
            print(f"[INFO] Overwriting existing entry for {name}.")
            existing_index = data["names"].index(name)
            data["encodings"][existing_index] = encodings[0] # Use the new encoding
        else:
            print("[INFO] Registration cancelled.")
            return False
    else:
         # Add the new encoding and name
        data["encodings"].append(encodings[0])
        data["names"].append(name)

    # Save the updated encodings
    print(f"[INFO] Saving encodings to {ENCODINGS_FILE}...")
    try:
        with open(ENCODINGS_FILE, "wb") as f:
            pickle.dump(data, f)
        print("[INFO] Encodings saved successfully.")
        # Optionally save a sample image
        save_sample_image(image, name)
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save encodings: {e}")
        return False

if __name__ == "__main__":
    # Get user's name
    user_name = input("Enter the name for the person: ").strip()
    if not user_name:
        print("[ERROR] Name cannot be empty.")
    else:
        # Capture image
        captured_image = capture_image_from_camera()

        if captured_image is not None:
            # Register the face
            success = register_new_face(user_name, captured_image)
            if success:
                print(f"[INFO] Registration complete for {user_name}.")
            else:
                print(f"[INFO] Registration failed for {user_name}.")
        else:
            print("[INFO] Registration process aborted.") 
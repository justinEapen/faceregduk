import cv2
import face_recognition
import pickle
import numpy as np
import time
import os
import RPi.GPIO as GPIO
from scipy.spatial import distance as dist # Added for EAR calculation
from collections import deque # Added for blink tracking

# --- Configuration ---
ENCODINGS_FILE = "encodings.pickle"
CAMERA_INDEX = 0 
# Model for face detection: 'hog' (faster on CPU) or 'cnn' (more accurate, GPU/TPU needed)
DETECTION_MODEL = 'hog' 
# How much to scale down the frame for faster processing (e.g., 0.5 = half size)
FRAME_RESIZE_SCALE = 0.5 
# Tolerance for face matching (lower is stricter). 0.6 is a good starting point.
FACE_MATCH_TOLERANCE = 0.6 
# How often (in seconds) to print detection results to avoid spamming
PRINT_INTERVAL = 2.0 
DISPLAY_VIDEO = True # Set to False to run headless (improves performance)

# --- Liveness Detection Configuration ---
# Threshold for eye aspect ratio to indicate blink
EYE_AR_THRESH = 0.25
# Number of consecutive frames the eye must be below the threshold for
EYE_AR_CONSEC_FRAMES = 3
# Number of frames to store for blink history per detected face
BLINK_HISTORY_LEN = 15
"""RELAY_PIN = 17  # Change this to the correct GPIO pin you're using
GPIO.setmode(GPIO.BCM)      # Use BCM pin numbering
GPIO.setup(RELAY_PIN, GPIO.OUT)  # Set pin as output"""
# --- End Configuration ---

# --- Helper Function for Liveness ---
def calculate_ear(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear
# --- End Helper Function ---

def load_encodings(encodings_path):
    """Loads face encodings and names from the pickle file."""
    print(f"[INFO] Loading known face encodings from {encodings_path}...")
    if not os.path.exists(encodings_path):
        print(f"[ERROR] Encodings file not found at {encodings_path}. Please run register_face.py first.")
        return None
        
    try:
        with open(encodings_path, "rb") as f:
            data = pickle.load(f)
        if not data["encodings"] or not data["names"]:
             print(f"[WARNING] Encodings file ({encodings_path}) is empty or corrupted.")
             return None
        print(f"[INFO] Loaded {len(data['names'])} known encodings.")
        return data
    except Exception as e:
        print(f"[ERROR] Could not load encodings file: {e}")
        return None

def main():
    # Load known faces
    known_data = load_encodings(ENCODINGS_FILE)
    if known_data is None:
        return

    known_encodings = known_data["encodings"]
    known_names = known_data["names"]

    # Initialize video stream
    print(f"[INFO] Starting video stream from camera {CAMERA_INDEX}...")
    vs = cv2.VideoCapture(CAMERA_INDEX)
    if not vs.isOpened():
        print(f"[ERROR] Cannot open camera {CAMERA_INDEX}.")
        return
        
    time.sleep(2.0) # Allow camera sensor to warm up

    print("[INFO] Starting face detection loop...")
    last_print_time = time.time()
    detected_name = "Unknown" # Keep track of the last detected name
    
    # --- Liveness Detection State ---
    # Store blink counters and history for faces (using simple index-based tracking for now)
    # For more robust multi-face tracking, a proper tracker (e.g., centroid tracking) would be needed.
    face_liveness = {} # Dictionary to store liveness state per face (key: face_index)
                       # value: {'blink_counter': int, 'blink_total': int, 'ear_history': deque, 'live': bool}
    # --- End Liveness State ---

    try:
        while True:
            # Grab the frame from the threaded video stream
            ret, frame = vs.read()
            if not ret:
                print("[WARNING] Could not read frame from camera. Trying to reconnect...")
                vs.release()
                time.sleep(5.0) # Wait before retrying
                vs = cv2.VideoCapture(CAMERA_INDEX)
                if not vs.isOpened():
                    print("[ERROR] Failed to reconnect to camera. Exiting.")
                    break
                continue

            # Resize frame for faster processing
            if FRAME_RESIZE_SCALE != 1.0:
                small_frame = cv2.resize(frame, (0, 0), fx=FRAME_RESIZE_SCALE, fy=FRAME_RESIZE_SCALE)
            else:
                small_frame = frame

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            # Detect the (x, y)-coordinates of the bounding boxes corresponding to each face in the input frame
            # Using the specified detection model
            boxes = face_recognition.face_locations(rgb_small_frame, model=DETECTION_MODEL)
            # Get facial landmarks BEFORE computing encodings (needed for EAR)
            landmarks_list = face_recognition.face_landmarks(rgb_small_frame, boxes)

            # --- Liveness Check ---
            new_face_liveness = {} # Track liveness for faces in the current frame
            live_indices = [] # Indices of faces deemed "live" in this frame

            for i, (box, landmarks) in enumerate(zip(boxes, landmarks_list)):
                # Initialize state for this face index if not seen before
                if i not in face_liveness:
                    face_liveness[i] = {
                        'blink_counter': 0,
                        'blink_total': 0, # Keep track of total blinks detected
                        'ear_history': deque(maxlen=BLINK_HISTORY_LEN),
                        'live': False # Start as not live until a blink is confirmed
                    }

                state = face_liveness[i]

                # Extract the left and right eye coordinates
                left_eye = landmarks['left_eye']
                right_eye = landmarks['right_eye']

                # Calculate the Eye Aspect Ratio (EAR) for both eyes
                left_ear = calculate_ear(left_eye)
                right_ear = calculate_ear(right_eye)

                # Average the EAR together for both eyes
                ear = (left_ear + right_ear) / 2.0
                state['ear_history'].append(ear)

                # Check to see if the eye aspect ratio is below the blink
                # threshold, and if so, increment the blink frame counter
                if ear < EYE_AR_THRESH:
                    state['blink_counter'] += 1
                # Otherwise, the eye aspect ratio is not below the blink threshold
                else:
                    # If the eyes were closed for a sufficient number of frames
                    # then increment the total number of blinks
                    if state['blink_counter'] >= EYE_AR_CONSEC_FRAMES:
                        state['blink_total'] += 1
                        state['live'] = True # Mark as live after the first blink
                        # Optional: Reset history after blink? Depends on desired behavior.
                        # state['ear_history'].clear() 

                    # Reset the eye frame counter
                    state['blink_counter'] = 0

                # Keep track of current state for this frame
                new_face_liveness[i] = state

                # If considered live based on history, add its index
                if state['live']:
                    live_indices.append(i)
            
            # Update the main liveness state (handle faces disappearing)
            face_liveness = new_face_liveness 
            # --- End Liveness Check ---

            # --- Face Recognition for Live Faces ---
            current_names = [] # Stores names/confidence for recognized LIVE faces

            # Loop over the indices of faces determined to be live
            for i in live_indices:
                # Get the box for this specific live face
                live_box = boxes[i]
                
                # Compute encoding for *this specific face*
                # face_recognition.face_encodings expects a list of known face locations (bounding boxes)
                # Note: This computes encodings one by one, which might be slightly less 
                # efficient than batching but avoids the index mapping issue.
                live_encoding_list = face_recognition.face_encodings(rgb_small_frame, [live_box]) 

                if not live_encoding_list:
                     # Should not happen if box was valid, but safety check
                     continue 

                encoding = live_encoding_list[0] # Get the single encoding computed

                # Attempt to match this face's encoding to our known encodings
                matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=FACE_MATCH_TOLERANCE)
                name = "Unknown"
                confidence = 0.0

                # Check to see if we have found a match
                if True in matches:
                    # Find the indexes of all matched faces then initialize a dictionary to count the total number of times each face was matched
                    matched_idxs = [j for (j, b) in enumerate(matches) if b]
                    counts = {}

                    # Loop over the matched indexes and calculate distances
                    face_distances = face_recognition.face_distance(known_encodings, encoding)
                    
                    best_match_index = -1
                    min_distance = 1.0 # Distances are typically <= 1.0

                    for k in matched_idxs: # Use k to avoid confusion with outer loop index i
                        distance = face_distances[k]
                        # Count successful matches for this name
                        matched_name = known_names[k]
                        counts[matched_name] = counts.get(matched_name, 0) + 1
                        
                        # Find the best match (lowest distance) among the matches
                        if distance < min_distance:
                           min_distance = distance
                           best_match_index = k

                    # Determine the recognized face with the largest number of votes (or best match if tied)
                    if counts:
                        # If there's a best match index (lowest distance), prioritize it
                        if best_match_index != -1:
                             name = known_names[best_match_index]
                             confidence = (1.0 - min_distance) * 100 # Convert distance to confidence %
                        else:
                             # Fallback if best_match_index wasn't set (shouldn't happen if matches=True)
                             name = max(counts, key=counts.get)
                             confidence = 70.0 # Arbitrary confidence

                # Append result, associating it with the original face index 'i'
                current_names.append({"name": name, "confidence": confidence, "box_index": i})
            # --- End Face Recognition ---


            # Process results only if PRINT_INTERVAL has passed or the detected name changes
            current_time = time.time()
            identified_person = next((p['name'] for p in current_names if p['name'] != "Unknown"), "Unknown")

            if (current_time - last_print_time > PRINT_INTERVAL) or (identified_person != detected_name and identified_person != "Unknown"):
                 
                 if identified_person != "Unknown" and confidence>70:
                      matched_person_info = next((p for p in current_names if p['name'] == identified_person), None)
                      conf_str = f"{matched_person_info['confidence']:.2f}%" if matched_person_info else "N/A"
                      print(f"[INFO] Detected: {identified_person} (Confidence: {conf_str})")
                      print("[SIGNAL] <<< True Signal / Face Detected >>>")
                      RELAY_PIN = 17  # Change this to the correct GPIO pin you're using

                      GPIO.setmode(GPIO.BCM)      # Use BCM pin numbering
                      GPIO.setup(RELAY_PIN, GPIO.OUT)  # Set pin as output

                      try:
                            print("Turning relay ON")
                            GPIO.output(RELAY_PIN, GPIO.HIGH)  # or GPIO.LOW depending on relay type
                            time.sleep(1)
                            print("Turning relay OFF")
                            GPIO.output(RELAY_PIN, GPIO.LOW)
                            time.sleep(1)
                      finally:
                            GPIO.cleanup()

                      # -----------------------------------------------------
                      # TODO: Add Door Unlocking Logic Here
                      # Example: activate_relay(), send_mqtt_signal(), etc.
                      # Be careful with security implications!
                      # -----------------------------------------------------
                      detected_name = identified_person # Update last known detected name
                 elif detected_name != "Unknown":
                     # If previously detected someone, but now only see "Unknown" or no one
                     print("[INFO] No known face detected.")
                     detected_name = "Unknown" # Reset detected name
                     
                 last_print_time = current_time


            # --- Display Logic (Optional) ---
            if DISPLAY_VIDEO:
                # Loop over the recognized faces to draw boxes
                # Note: boxes are relative to the resized frame, scale them back up
                upscale = 1.0 / FRAME_RESIZE_SCALE if FRAME_RESIZE_SCALE != 0 else 1.0
                
                # Draw boxes and status for ALL detected faces (live or not)
                for i, (top, right, bottom, left) in enumerate(boxes):
                     # Scale box back to original frame size
                     top, right, bottom, left = int(top * upscale), int(right * upscale), int(bottom * upscale), int(left * upscale)
                     
                     # Default color and label for non-live/unknown
                     color = (0, 0, 255) # Red for not live / unknown
                     label = "Checking Liveness..."
                     
                     is_live = face_liveness.get(i, {}).get('live', False)
                     
                     # Check if this face (by index) was recognized
                     recognized_face_info = next((p for p in current_names if p.get('box_index') == i), None)

                     if recognized_face_info:
                         # It's a recognized live face
                         color = (0, 255, 0) # Green for recognized
                         label = f"{recognized_face_info['name']} ({recognized_face_info['confidence']:.1f}%)"
                     elif is_live:
                         # It's live, but not recognized
                         color = (255, 255, 0) # Cyan for live but unknown
                         label = "Unknown (Live)"

                     # Draw a box around the face
                     cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                     # Draw a label with a name/status below the face
                     y = top - 15 if top - 15 > 15 else top + 15
                     cv2.putText(frame, label, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Display the resulting image
                cv2.imshow("Face Recognition Security Feed", frame)

                # Break loop if 'q' is pressed
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("[INFO] Quitting detection loop.")
                    break
            # --- End Display Logic ---

    except KeyboardInterrupt:
        print("[INFO] Keyboard interrupt received. Stopping...")
    finally:
        # Clean up
        print("[INFO] Releasing resources...")
        vs.release()
        if DISPLAY_VIDEO:
             cv2.destroyAllWindows()
             # Add small delay for windows to close properly
             for i in range(5):
                 cv2.waitKey(1)
        print("[INFO] System stopped.")


if __name__ == "__main__":
    main() 

# Raspberry Pi Face Recognition Security System

This project implements a two-part face recognition system for a Raspberry Pi:

1.  **`register_face.py`**: A utility to capture faces using the Pi camera, associate them with names, and store their encodings locally.
2.  **`detect_face.py`**: The main security application that continuously monitors the camera feed, compares detected faces against the stored encodings, and signals when a recognized face is found.

## Features

*   Local storage of face encodings (no external database needed).
*   Configurable face detection model (`hog` for CPU, `cnn` for GPU/TPU).
*   Adjustable frame resizing for performance tuning.
*   Adjustable face matching tolerance.
*   Confidence percentage display for recognized faces.
*   Clear placeholder for integrating door unlocking logic.
*   Optional live video feed display with detected faces and names.
*   Error handling for camera issues.
*   Optimized for continuous operation on Raspberry Pi.

## Installation

1.  **Clone the repository (or download the files):**
    ```bash
    # git clone <repository_url> # If using git
    # cd <repository_directory>
    ```

2.  **Install Dependencies:**
    It's highly recommended to use a Python virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate # On Linux/macOS
    # venv\Scripts\activate # On Windows
    
    pip install -r requirements.txt
    ```

    **Important Notes for Raspberry Pi:**
    *   Installing `dlib` (a dependency of `face_recognition`) on Raspberry Pi can be time-consuming and memory-intensive as it often requires compilation from source.
    *   Ensure your Raspberry Pi has enough RAM and **sufficient swap space allocated** before installation. You might need to increase swap size temporarily:
        ```bash
        # Check current swap
        free -h 
        swapon --show
        
        # Increase swap (Example: increase to 2GB)
        sudo dphys-swapfile swapoff
        sudo sed -i 's/CONF_SWAPSIZE=.*/CONF_SWAPSIZE=2048/' /etc/dphys-swapfile
        sudo dphys-swapfile setup
        sudo dphys-swapfile swapon
        
        # Verify new swap size
        free -h
        ```
    *   The installation might take **over an hour** on older Pi models.
    *   If `pip install dlib` fails, you might need to install prerequisites first:
        ```bash
        sudo apt-get update
        sudo apt-get install build-essential cmake
        sudo apt-get install libopenblas-dev liblapack-dev 
        sudo apt-get install libx11-dev libgtk-3-dev # For GUI components if needed
        # Then try pip install again
        ```
    *   Remember to **reduce swap size back** to a normal level (e.g., 100MB or default) after installation to avoid excessive wear on the SD card:
        ```bash
        sudo dphys-swapfile swapoff
        sudo sed -i 's/CONF_SWAPSIZE=.*/CONF_SWAPSIZE=100/' /etc/dphys-swapfile # Or your preferred default
        sudo dphys-swapfile setup
        sudo dphys-swapfile swapon
        ```

## Usage

1.  **Register Faces:**
    *   Run the registration script:
        ```bash
        python register_face.py
        ```
    *   Enter the name for the person when prompted.
    *   A camera window will open. Position the face clearly in the frame.
    *   Press `c` to capture the image.
    *   Press `q` to quit without capturing.
    *   The script will process the image, extract the encoding, and save it to `encodings.pickle`.
    *   Repeat for each person you want to authorize.
    *   Captured sample images are optionally saved in the `image_samples/` directory.

2.  **Run the Detection System:**
    *   Ensure `encodings.pickle` exists (created by the registration script).
    *   Run the detection script:
        ```bash
        python detect_face.py
        ```
    *   The system will start monitoring the camera feed.
    *   If a registered face is detected with sufficient confidence (based on `FACE_MATCH_TOLERANCE`), it will print the person's name, confidence, and the "True Signal / Face Detected" message to the console.
    *   If `DISPLAY_VIDEO` is `True` (default), a window will show the camera feed with bounding boxes and names.
    *   Press `q` in the display window or `Ctrl+C` in the terminal to stop the system.

## Configuration (`detect_face.py`)

You can adjust the following parameters at the top of the `detect_face.py` script:

*   `ENCODINGS_FILE`: Path to the saved encodings file.
*   `CAMERA_INDEX`: Index of the camera to use (usually 0).
*   `DETECTION_MODEL`: `'hog'` (faster, less accurate) or `'cnn'` (slower, more accurate, requires dlib compiled with CUDA for GPU acceleration).
*   `FRAME_RESIZE_SCALE`: Factor to resize frames for faster processing (e.g., `0.5` for half size). `1.0` uses original size.
*   `FACE_MATCH_TOLERANCE`: Threshold for recognizing a face (lower is stricter, `0.6` is a common default).
*   `PRINT_INTERVAL`: Minimum time (seconds) between printing detection results to avoid console spam.
*   `DISPLAY_VIDEO`: `True` to show the live video feed, `False` to run headless (better performance).

## TODO / Integration

*   The `detect_face.py` script has a marked `TODO` section where you can add code to interact with hardware (like relays, GPIO pins) to unlock a door when a valid face is detected. 
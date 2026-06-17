# Live Object Detection

live-object-detection is a small Python computer vision prototype that runs YOLOv8 object detection on a live webcam feed. It opens the default camera, runs frame-by-frame detection, and displays annotated results in an OpenCV window.

## Features

- Loads a pretrained YOLOv8 nano model.
- Captures frames from the default webcam.
- Runs real-time object detection with a configurable confidence threshold.
- Displays annotated detection output in an OpenCV window.
- Includes optional camera resolution settings in the script.

## Tech Stack

- Python
- Ultralytics YOLOv8
- OpenCV
- PyTorch runtime dependencies

## Project Structure

- yolo_webcam.py - webcam capture and detection loop
- requirements.txt - Python dependencies

## Getting Started

Create a virtual environment, install dependencies, and run the script:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python yolo_webcam.py
```

Press Q in the OpenCV window to quit.

## Status

Focused computer vision prototype. This is a concise demo repo rather than a full application.

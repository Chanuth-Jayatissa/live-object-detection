from ultralytics import YOLO
import cv2

# Load YOLOv8 model pretrained on COCO (80 classes)
model = YOLO("yolov8n.pt")  # n = nano (fast), use yolov8s.pt for slightly better accuracy

# Start webcam (0 = default laptop camera)
cap = cv2.VideoCapture(0)

# Uncomment and change the below two lines to set a specific camera resolution (if not it sets to 640x480 by default)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Print camera resolution
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"Camera resolution: {width} x {height}")


while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip camera
    frame = cv2.flip(frame, 1)
    
    # crop camera horizontally to make it look like normal camera app
    h, w, _ = frame.shape
    crop_x = int(w * 0.1)  # Crop 10% from the left
    frame = frame[:, crop_x:-crop_x]  # Crop the frame

    # Run object detection
    results = model.predict(source=frame, conf=0.4, verbose=False)

    # Plot detection results on the frame
    annotated_frame = results[0].plot()

    # Show in window
    cv2.imshow("YOLOv8 Webcam Detection", annotated_frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

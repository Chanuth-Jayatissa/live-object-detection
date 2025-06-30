from ultralytics import YOLO
import cv2

# Load YOLOv8 model pretrained on COCO (80 classes)
model = YOLO("yolov8n.pt")  # n = nano (fast), use yolov8s.pt for slightly better accuracy

# Start webcam (0 = default laptop camera)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

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

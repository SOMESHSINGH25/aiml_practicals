
import cv2
from ultralytics import YOLO
import os

# Load YOLO11 nano model
model = YOLO("yolo11n.pt")

# Open CCTV video
video_path = "cctv.mp4"

if not os.path.exists(video_path):
    print("Error: cctv.mp4 not found in the current folder.")
    raise SystemExit

video = cv2.VideoCapture(video_path)

if not video.isOpened():
    print("Error: Unable to open CCTV video.")
    raise SystemExit

print("CCTV Object Detection Started")
print("Press Q to exit.")

while True:
    ret, frame = video.read()

    if not ret:
        break

    # Detect objects using CPU
    results = model.predict(
        source=frame,
        device="cpu",
        imgsz=640,
        verbose=False
    )

    # Draw bounding boxes and labels
    annotated_frame = results[0].plot()

    # Display detected objects
    cv2.imshow("CCTV Object Detection", annotated_frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
video.release()
cv2.destroyAllWindows()

print("Object detection completed.")

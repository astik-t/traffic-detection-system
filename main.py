from ultralytics import YOLO
import cv2

# Load model (GPU automatically used)
model = YOLO("yolov8n.pt")

# COCO class IDs
vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck

def traffic_level(vehicle_count):
    if vehicle_count <= 5:
        return "Low"
    elif vehicle_count <= 15:
        return "Moderate"
    else:
        return "Heavy"

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, device=0)

    vehicle_count = 0

    for box in results[0].boxes:
        cls = int(box.cls[0])
        if cls in vehicle_classes:
            vehicle_count += 1

    level = traffic_level(vehicle_count)

    annotated_frame = results[0].plot()

    cv2.putText(annotated_frame, f"Vehicles: {vehicle_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.putText(annotated_frame, f"Traffic: {level}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("Traffic Monitoring", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
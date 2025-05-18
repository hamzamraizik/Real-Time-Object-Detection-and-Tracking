import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort

# Initialize YOLOv8 model (YOLOv8n is fast; change to 'yolov8s.pt' or others as needed)
model = YOLO('yolov8n.pt')

# Initialize SORT tracker
tracker = Sort()

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv8 expects RGB
    results = model(frame[..., ::-1])
    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            # SORT expects [x1, y1, x2, y2, score]
            detections.append([x1, y1, x2, y2, conf])
    dets = np.array(detections)
    tracks = tracker.update(dets)

    # Draw boxes and track IDs
    for track in tracks:
        x1, y1, x2, y2, track_id = track.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f'ID {track_id}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow('YOLOv8 + SORT Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 
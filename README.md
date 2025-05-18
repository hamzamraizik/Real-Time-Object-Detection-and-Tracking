# Real-Time Object Detection and Tracking

This project uses YOLOv8 for object detection and SORT for tracking in real-time video streams (webcam).

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Download a YOLOv8 model (e.g., `yolov8n.pt`) from [Ultralytics YOLOv8 Releases](https://github.com/ultralytics/ultralytics/releases).
   Place it in the project directory or specify the path in the script.

## Usage

Run the main script:
```bash
python object_detection_tracking.py
```

- Press `q` to quit.

## Notes
- The script uses your webcam by default. To use a video file, change the `cv2.VideoCapture(0)` line to `cv2.VideoCapture('your_video.mp4')`.
- You can switch to a different YOLOv8 model (e.g., `yolov8s.pt`) for better accuracy or speed. 
# Motion-Gated Person Detection

This project uses OpenCV to detect motion in a live camera feed and runs a
YOLO11 model when motion is present. Person detections are confirmed across
multiple model results before a console alert is printed. The alert function is
a placeholder; it does not send Telegram messages yet.

## Run

Install the Python packages:

```powershell
python -m pip install ultralytics opencv-python
```

Run from the `videostream` folder so the model file is found:

```powershell
cd videostream
python waking_yolo_by_motion.py
```

Press **q** in the video window to quit. A webcam is used by default; edit
`SOURCE` in the script to use a video file or stream URL.

## Other script

`videostream/motion_detection.py` shows motion detection on its own, without
YOLO. The `yolo_testing` folder contains scripts and images for testing YOLO.

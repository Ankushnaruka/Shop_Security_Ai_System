import time
import threading
from collections import deque

import cv2
from ultralytics import YOLO

# ---------------- Settings ----------------
SOURCE        = 0          # 0 = webcam, or RTSP/HTTP URL, or "test.mp4"
TARGET_FPS    = 8          # processing rate (5-10)
MIN_AREA      = 500        # motion blob size filter
HOLD_SECONDS  = 5          # keep running YOLO this long after last motion
HEARTBEAT_S   = 3          # run YOLO at least this often even without motion
CONF          = 0.25
IMGSZ         = 960
VOTE_WINDOW   = 8          # look at last N YOLO results
VOTE_NEEDED   = 3          # person must appear in this many of them
WARMUP_FRAMES = 30         # let MOG2 learn the background first
# ------------------------------------------

class LatestFrameReader:
    """Reads the camera in a background thread and keeps only the newest frame,
    so processing never works on stale, buffered frames."""
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open video source")
        self.frame = None
        self.ok = True
        self.lock = threading.Lock()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.ok:
            ok, f = self.cap.read()
            if not ok:
                self.ok = False      # stream ended/dropped (add reconnect later)
                break
            with self.lock:
                self.frame = f

    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.ok = False
        self.cap.release()


def detect_motion(frame, subtractor, kernel):
    gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (21, 21), 0)
    mask = subtractor.apply(gray)
    _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=3)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= MIN_AREA]
    return boxes


def on_person_alert(frame):
    """Placeholder: Telegram alert goes here later."""
    print(f"[{time.strftime('%H:%M:%S')}] PERSON DETECTED")


def main():
    reader = LatestFrameReader(SOURCE)
    model = YOLO("yolo11m.pt")
    subtractor = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=25, detectShadows=True)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    votes = deque(maxlen=VOTE_WINDOW)
    last_motion = 0.0
    last_yolo = 0.0
    frame_count = 0
    person_boxes = []
    frame_interval = 1.0 / TARGET_FPS
    next_time = time.time()

    while reader.ok:
        # --- throttle to TARGET_FPS ---
        now = time.time()
        if now < next_time:
            time.sleep(0.002)
            continue
        next_time = now + frame_interval

        frame = reader.read()
        if frame is None:
            continue
        frame_count += 1

        # --- 1. motion gate (runs on every processed frame) ---
        motion_boxes = detect_motion(frame, subtractor, kernel)
        if frame_count < WARMUP_FRAMES:
            motion_boxes = []                    # ignore noise while learning
        if motion_boxes:
            last_motion = now

        # --- 2. decide whether to run YOLO ---
        in_hold   = (now - last_motion) < HOLD_SECONDS
        heartbeat = (now - last_yolo) > HEARTBEAT_S
        run_yolo  = frame_count >= WARMUP_FRAMES and (in_hold or heartbeat)

        if run_yolo:
            last_yolo = now
            results = model.predict(frame, conf=CONF, imgsz=IMGSZ,
                                    classes=[0], verbose=False)   # 0 = person
            person_boxes = [tuple(map(int, b.xyxy[0])) for b in results[0].boxes]
            votes.append(len(person_boxes) > 0)

            if sum(votes) >= VOTE_NEEDED:
                on_person_alert(frame)
                votes.clear()                    # avoid re-firing every frame

        # --- 3. draw ---
        for (x, y, w, h) in motion_boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 1)    # yellow = motion
        for (x1, y1, x2, y2) in person_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)          # green = person
        status = "YOLO ON" if run_yolo else "YOLO idle"
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("Live", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    reader.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
import cv2

cap = cv2.VideoCapture(0)
subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500,        # how many frames to learn the background from
    varThreshold=15,    # lower = more sensitive
    detectShadows=True,
)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
MIN_AREA = 100          # tune this for your camera

while True:
    ok, frame = cap.read()
    if not ok:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    mask = subtractor.apply(gray)
    _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)  # drop shadows
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=3)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    motion = False
    for c in contours:
        if cv2.contourArea(c) < MIN_AREA:
            continue
        motion = True
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.putText(frame, "MOTION" if motion else "Idle", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 0, 255) if motion else (0, 255, 0), 2)

    cv2.imshow("Live", frame)
    cv2.imshow("Mask", mask)    # keep this while tuning
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
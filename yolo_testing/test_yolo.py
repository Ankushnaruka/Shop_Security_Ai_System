from ultralytics import YOLO

# Load a pretrained YOLO model
model = YOLO("yolo11m.pt")

# Test it on an image
results = model.predict(
    source="test.jpg",
    conf=0.3 # Set the confidence threshold for detection
)

# Display the image with detected objects
results[0].show()

# Save the output image with bounding boxes
results[0].save(filename="output.jpg")

# Print detected objects
for box in results[0].boxes:
    class_id = int(box.cls[0])
    confidence = float(box.conf[0])
    class_name = model.names[class_id]
    print(f"{class_name}: {confidence:.2f}")

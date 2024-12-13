import cv2
import torch
import numpy as np

# Load the pre-trained YOLOv5 model from Ultralytics
model = torch.hub.load('ultralytics/yolov5', 'yolov5l')  # You can choose 'yolov5m', 'yolov5l', or 'yolov5x' based on your needs

# Open the webcam (0 is the default camera index)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Failed to capture image.")
        break
    
    # Perform inference on the current frame
    results = model(frame)
    
    # Get bounding boxes in xyxy format
    boxes = results.xyxy[0].numpy()  # Bounding boxes are returned in [x_min, y_min, x_max, y_max, confidence, class_id]
    
    # Loop over the bounding boxes
    for box in boxes:
        x_min, y_min, x_max, y_max, conf, class_id = box
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        
        # Draw the bounding box and center point
        cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)  # Green box
        cv2.circle(frame, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)  # Red circle at the center
        
        # Optionally, put the class name and confidence on the bounding box
        label = f"{model.names[int(class_id)]} {conf:.2f}"
        cv2.putText(frame, label, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        center_text = f"({int(center_x)}, {int(center_y)})"
        
        cv2.putText(frame, center_text, (int(center_x) + 10, int(center_y) - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)  # Yellow color for center coordinates
    
    # Show the frame with bounding boxes and centers
    cv2.imshow('YOLOv5 Object Detection', frame)

    # Break the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture object and close the OpenCV window
cap.release()
cv2.destroyAllWindows()

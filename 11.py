import cv2
import numpy as np
from ultralytics import YOLO
import torch

def init_model():
    """Initialize the YOLO model"""
    try:
        # Load the YOLOv5 model
        model = YOLO('best.pt')
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def init_camera():
    """Initialize the camera"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None
    return cap

def process_frame(frame, model):
    """Process a single frame with object detection"""
    try:
        # Perform inference
        results = model(frame, verbose=False)
        
        # Process detections
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Calculate center
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                # Draw bounding box (green)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw center point (red)
                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
                
                # Add labels
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = f"{model.names[cls]} {conf:.2f}"
                
                # Draw label (white)
                cv2.putText(frame, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Draw center coordinates (yellow)
                center_text = f"({center_x}, {center_y})"
                cv2.putText(frame, center_text, (center_x + 10, center_y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        return frame
    except Exception as e:
        print(f"Error processing frame: {e}")
        return frame

def main():
    # Initialize model
    model = init_model()
    if model is None:
        return

    # Initialize camera
    cap = init_camera()
    if cap is None:
        return

    print("Object detection started. Press 'q' to quit.")

    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image.")
                break

            # Process frame
            processed_frame = process_frame(frame, model)

            # Display frame
            cv2.imshow('YOLOv5 Object Detection', processed_frame)

            # Check for quit command
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Quitting...")
                break

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
import cv2
import torch
import pyttsx3
from vosk import Model, KaldiRecognizer
import numpy as np
import wave 
from ultralytics import YOLO
import pyaudio
import json
model_path = "/Users/nikhil/Downloads/vosk-model-small-en-us-0.15"  # Update this path with your model directory


vosk_model = Model(model_path)
recognizer = KaldiRecognizer(vosk_model, 16000)

def init_model():
    """Initialize the YOLO model"""
    try:
        # Load the YOLOv5 model
        model = YOLO('/Users/nikhil/Downloads/best (1).pt')
        model.conf = 0.25
        model.iou = 0.45
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

def listen_for_command():
    """
    Listen for voice command using Vosk
    """
    print("Listening for command...")
    p = None
    stream = None
    try:
        p = pyaudio.PyAudio()
        
        # List available input devices
        info = p.get_host_api_info_by_index(0)
        numdevices = info.get('deviceCount')
        
        # Find the first available input device
        input_device_index = None
        for i in range(numdevices):
            if p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels') > 0:
                input_device_index = i
                break
        
        if input_device_index is None:
            print("No input devices found!")
            return None
            
        # Configure and open stream with specific device
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            input_device_index=input_device_index,
            frames_per_buffer=8192
        )
        
        stream.start_stream()
        
        try:
            print("Recording audio...")
            while True:
                data = stream.read(4096, exception_on_overflow=False)
                if recognizer.AcceptWaveform(data):
                    result_json = recognizer.Result()
                    # Parse the JSON string and extract the text
                    result_dict = json.loads(result_json)
                    result = result_dict.get('text', '')
                    stream.stop_stream()
                    stream.close()
                    p.terminate()
                    return result
                    
        except Exception as e:
            print(f"Error recording audio: {e}")
            return None
            
    except Exception as e:
        print(f"Error initializing audio: {e}")
        return None
    finally:
        if stream is not None and stream.is_active():
            stream.stop_stream()
            stream.close()
        if p is not None:
            p.terminate()
        
        
def process_frame(frame, model):
    """Process a single frame with object detection"""
    try:
        # Perform inference
        results = model(frame, verbose=False)
        
        # Process detections
        detected_objects = []
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
                detected_objects.append({
                    'class': model.names[int(cls)],
                    'center': (int(center_x), int(center_y)),
                    'confidence': conf  
                })
                
                # Draw label (white)
                cv2.putText(frame, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Draw center coordinates (yellow)
                center_text = f"({center_x}, {center_y})"
                cv2.putText(frame, center_text, (center_x + 10, center_y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        return frame, detected_objects
    except Exception as e:
        print(f"Error processing frame: {e}")
        return frame, detected_objects



# Initialize model
model = init_model()
if model is None:
    print("Failed to load model.")

# Initialize camera
cap = init_camera()
if cap is None:
    print("Failed to initialize camera.")
   

print("Object detection started. Press 'q' to quit.")

try:
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break
        # Process frame
        processed_frame, detected_objects = process_frame(frame, model)

        # Display frame
        cv2.imshow('YOLOv5 Object Detection', processed_frame)

        command = listen_for_command()
        if command:
            print(f"Command recognized: {command}")
            for obj in detected_objects:
                if command.lower() in obj['class'].lower():
                    print(f"Selected object: {obj['class']} at center point {obj['center']}")
                    
                else: 
                    print("No matching object found for the command.")

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



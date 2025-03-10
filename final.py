import cv2
import torch
import pyttsx3
from vosk import Model, KaldiRecognizer
import numpy as np
import wave 
from ultralytics import YOLO
import pyaudio
import json
import time

model_path = "/Users/evansvetina/Downloads/vosk-model-small-en-us-0.15"  # Update this path with your model directory

vosk_model = Model(model_path)
recognizer = KaldiRecognizer(vosk_model, 16000)

# Global variable to store detected objects between frames
detected_objects = []

def init_model():
    """Initialize the YOLO model"""
    try:
        # Load the YOLOv5 model
        model = YOLO('/Users/evansvetina/Downloads/best (1).pt')
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
    Listens for up to 3 seconds for a command
    """
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
            frames_per_buffer=4096
        )
        
        stream.start_stream()
        
        # Set maximum listening time to 3 seconds
        start_time = time.time()
        max_listen_time = 3  # Listen for up to 3 seconds
        
        try:
            while time.time() - start_time < max_listen_time:
                data = stream.read(4096, exception_on_overflow=False)
                
                if recognizer.AcceptWaveform(data):
                    result_json = recognizer.Result()
                    # Parse the JSON string and extract the text
                    result_dict = json.loads(result_json)
                    result = result_dict.get('text', '')
                    if result:  # Only return if we got an actual command
                        return result
            
            # If we reach here, we didn't get a command in the time allotted
            return None
                    
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
    global detected_objects  # Make this a global variable to persist between frames
    
    # Only reset the detected objects list if new objects are found
    current_detected_objects = []
    
    try:
        # Check if model is None
        if model is None:
            return frame, detected_objects
            
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
                class_name = model.names[cls]
                label = f"{class_name} {conf:.2f}"
                
                # Store the detected object
                current_detected_objects.append({
                    'class': class_name,
                    'center': (center_x, center_y),
                    'confidence': conf  
                })
                
                # Draw label (white)
                cv2.putText(frame, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Draw center coordinates (yellow)
                center_text = f"({center_x}, {center_y})"
                cv2.putText(frame, center_text, (center_x + 10, center_y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Only update detected_objects if we found objects
        if current_detected_objects:
            detected_objects = current_detected_objects
        
        return frame, detected_objects
    except Exception as e:
        return frame, detected_objects  # Return the frame and empty list


# Initialize model
model = init_model()

# Initialize camera
cap = init_camera()

# Initialize the speech recognizer once to avoid reopening it repeatedly

   
# Use time-based intervals instead of frame-based intervals
import time
last_voice_check_time = time.time()
VOICE_CHECK_INTERVAL_SECONDS = 5  # Check for voice commands every 5 seconds


try:
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame
        processed_frame, detected_objects = process_frame(frame, model)

        # Display frame
        cv2.imshow('YOLOv5 Object Detection', processed_frame)

        # Check for voice commands every 5 seconds
        current_time = time.time()
        if current_time - last_voice_check_time >= VOICE_CHECK_INTERVAL_SECONDS:
            last_voice_check_time = current_time
            
            # Check for voice commands
            command = listen_for_command()
            
            # Process command if we got one
            if command:
                found_match = False
                # Convert command to lowercase for case-insensitive comparison
                command_lower = command.lower().strip()
                
                # Try a more flexible matching approach
                for obj in detected_objects:
                    # Get the object class and make it lowercase
                    obj_class = obj['class'].lower().strip()
                    
                    # Check for partial matches in either direction
                    if (command_lower in obj_class) or (obj_class in command_lower) or \
                       ('bandage' in command_lower and 'bandage' in obj_class) or \
                       ('scissor' in command_lower and 'scissor' in obj_class):
                        
                        # Format the center coordinates as x,y,z with y always 0
                        center_x, center_y = obj['center']
                        print(f"{center_x},0,{center_y}")
                        found_match = True
                        break  # Exit after first match
                
                if not found_match:
                    print(f"No matching object found for the command '{command}'.")
        
        # Process key presses
        key = cv2.waitKey(1) & 0xFF
        
        # Check for quit command
        if key == ord('q'):
            break

except Exception as e:
    pass

finally:
    # Clean up
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
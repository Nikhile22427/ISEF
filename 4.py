import cv2
import torch
import numpy as np
import pyttsx3
import os
import wave
from vosk import Model, KaldiRecognizer
import pyaudio
import yaml
import sys
from pathlib import Path

def setup_yolov5():
    """
    Setup YOLOv5 by cloning the repository if it doesn't exist
    """
    YOLOV5_PATH = "yolov5"  # Directory name for YOLOv5
    
    if not os.path.exists(YOLOV5_PATH):
        # Clone YOLOv5 repository
        os.system("git clone https://github.com/ultralytics/yolov5.git")
        
    # Add YOLOv5 to path
    if YOLOV5_PATH not in sys.path:
        sys.path.append(YOLOV5_PATH)

def load_custom_yolov5m(weights_path, data_yaml_path):
    """
    Load YOLOv5m model with custom weights
    """
    try:
        # Setup YOLOv5
        setup_yolov5()
        
        # Import YOLOv5 modules (after adding to path)
        from models.experimental import attempt_load
        
        # Load the model weights
        device = torch.device('cpu')
        model = attempt_load(weights_path, device=device)
        
        # Load class names from data.yaml
        with open(data_yaml_path, 'r') as f:
            data_yaml = yaml.safe_load(f)
            class_names = data_yaml['names']
            
        # Update class names in the model
        model.names = class_names
        
        # Set model to evaluation mode
        model.eval()
        
        print("Successfully loaded YOLOv5m with custom weights")
        return model
        
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

# First, make sure you have the required packages
print("Installing required packages...")
os.system("pip install -r yolov5/requirements.txt")

# Initialize model with custom weights
weights_path = "/Users/nikhil/Downloads/best.pt"  # Update this path
data_yaml_path = "/Users/nikhil/Downloads/ISEF training dataset.v1i.yolov5pytorch (1)/data.yaml"  # Update this path

try:
    model = load_custom_yolov5m(weights_path, data_yaml_path)
    
    # Set model parameters
    model.conf = 0.25  # Confidence threshold
    model.iou = 0.45   # NMS IOU threshold
    
    print(f"Model loaded successfully on: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    
except Exception as e:
    print(f"Error initializing model: {e}")
    exit()

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Load Vosk model for speech recognition
vosk_model_path = "/Users/nikhil/Downloads/vosk-model-small-en-us-0.15"
if not os.path.exists(vosk_model_path):
    print("Error: Vosk model path is invalid.")
    exit()

vosk_model = Model(vosk_model_path)
recognizer = KaldiRecognizer(vosk_model, 16000)


# Open the webcam
cap = cv2.VideoCapture(0, cv2.CAP_ANY)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

def listen_for_command():
    print("Listening for command...")
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8192)
    stream.start_stream()

    try:
        print("Recording audio...")
        while True:
            data = stream.read(4096, exception_on_overflow=False)
            if recognizer.AcceptWaveform(data):
                result = recognizer.Result()
                return result
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

# Main detection loop
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Convert frame for YOLOv5m
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        convertedframe_rgb = cv2.resize(frame_rgb, (640, 480))
        convertedframe_rgb = torch.from_numpy(convertedframe_rgb).permute(2, 0, 1).unsqueeze(0).to(device='cpu').float()
        # Perform inference with YOLOv5m
        results = model(convertedframe_rgb)  # YOLOv5 inference
        results = results[0]  # Extract first output tensor
        boxes = results.cpu().numpy()  # Convert to numpy
        detected_objects = []
        for box in boxes:
            # Check if the box contains the expected number of values
            if len(box) == 46:  # Ensure there are at least 46 values (bounding box + class probabilities)
                x_min, y_min, x_max, y_max, conf = box[:5]  # Extract bounding box and confidence
                x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)  # Convert to integers
                # Class probabilities are from index 5 to the end
                class_probabilities = box[5:]
                
                # Get the class ID with the highest probability
                class_id = int(np.argmax(class_probabilities))  # Get the index of the max class probability

                # Calculate center coordinates
                center_x = (x_min + x_max) / 2
                center_y = (y_min + y_max) / 2
                
                detected_objects.append({
                    'class': model.names[class_id],  # Get class name using class ID
                    'center': (int(center_x), int(center_y)),  # Store center coordinates
                    'confidence': conf
                })
            else:
                print(f"Warning: Box has an unexpected number of values (only {len(box)}). Skipping box.")



            # Draw bounding box and center point
            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
            cv2.circle(frame, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)

            # Add labels
            label = f"{model.names[int(class_id)]} {conf:.2f}"
            cv2.putText(frame, label, (int(x_min), int(y_min) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            center_text = f"({int(center_x)}, {int(center_y)})"
            cv2.putText(frame, center_text, (int(center_x) + 10, int(center_y) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        cv2.imshow('YOLOv5m Custom Object Detection', frame)

        command = listen_for_command()

        if command:
            print(f"Command recognized: {command}")
            for obj in detected_objects:
                if obj['class'].lower() in command.lower():
                    print(f"Selected object: {obj['class']} at center point {obj['center']}")
                    engine.say(f"The center of the {obj['class']} is at {obj['center']}")
                    engine.runAndWait()
                    break
            else:
                print("No matching object found for the command.")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"Error during execution: {e}")

finally:
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
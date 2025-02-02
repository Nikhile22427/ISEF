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

def listen_for_command():
    """
    Listen for voice command using Vosk
    """
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

# First, make sure you have the required packages
print("Installing required packages...")
os.system("pip install -r yolov5/requirements.txt")

# Initialize model with custom weights
weights_path = "/Users/nikhil/Downloads/best.pt"  # Update this path
data_yaml_path = "/Users/nikhil/Downloads/ISEF training dataset.v1i.yolov5pytorch (1)/data.yaml"  # Update this path

# Initialize the model before the camera loop
print("Loading YOLOv5 model...")
model = load_custom_yolov5m(weights_path, data_yaml_path)

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Initialize speech recognition
model_path = "/Users/nikhil/Downloads/vosk-model-small-en-us-0.15"  # Update this with your Vosk model path
if not os.path.exists(model_path):
    print("Please download a model for your language from https://alphacephei.com/vosk/models")
    sys.exit(1)
    
vosk_model = Model(model_path)
recognizer = KaldiRecognizer(vosk_model, 16000)

# Initialize camera
cap = cv2.VideoCapture(0)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Convert frame for YOLOv5m
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        convertedframe_rgb = cv2.resize(frame_rgb, (640, 480))
        convertedframe_rgb = torch.from_numpy(convertedframe_rgb).permute(2, 0, 1).unsqueeze(0).to(device='cpu').float() / 255.0

        # Perform inference with YOLOv5m
        with torch.no_grad():
            results = model(convertedframe_rgb)
        
        # Process predictions
        detected_objects = []
        
        # Get the predictions
        preds = results[0]  # Get first batch
        if isinstance(preds, torch.Tensor):  # Make sure we have a tensor
            # Convert to numpy array
            try:
                detections = preds.cpu().numpy()
                
                # Check if we have any detections
                if detections.shape[0] > 0:
                    # Iterate through detections
                    for detection in detections:
                        try:
                            # Extract scores (assuming format matches YOLOv5)
                            scores = detection[4:]  # Get confidence scores
                            class_id = np.argmax(scores)  # Get class with highest score
                            confidence = scores[class_id]  # Get the confidence value
                            
                            # Only process if confidence is above threshold
                            if confidence >= model.conf:
                                # Get box coordinates (assuming YOLO format: x, y, w, h)
                                x = float(detection[0])  # center x
                                y = float(detection[1])  # center y
                                w = float(detection[2])  # width
                                h = float(detection[3])  # height
                                
                                # Convert to pixel coordinates
                                img_height, img_width = frame.shape[:2]
                                x = int(x * img_width)
                                y = int(y * img_height)
                                w = int(w * img_width)
                                h = int(h * img_height)
                                
                                # Calculate box corners
                                x1 = max(0, x - w//2)
                                y1 = max(0, y - h//2)
                                x2 = min(img_width, x + w//2)
                                y2 = min(img_height, y + h//2)
                                
                                # Store detection
                                detected_objects.append({
                                    'class': model.names[class_id],
                                    'center': (x, y),
                                    'confidence': float(confidence)
                                })

                                # Draw bounding box
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

                                # Add labels
                                label = f"{model.names[class_id]} {confidence:.2f}"
                                cv2.putText(frame, label, (x1, y1 - 10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                                
                                center_text = f"({x}, {y})"
                                cv2.putText(frame, center_text, (x + 10, y - 10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                        
                        except Exception as e:
                            print(f"Error processing individual detection: {e}")
                            continue
                            
            except Exception as e:
                print(f"Error converting predictions: {e}")
                continue

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
    import traceback
    traceback.print_exc()

finally:
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
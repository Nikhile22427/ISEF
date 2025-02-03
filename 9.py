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
import json
from pathlib import Path

def setup_yolov5():
    """
    Setup YOLOv5 by cloning the repository if it doesn't exist
    """
    YOLOV5_PATH = "yolov5"
    
    if not os.path.exists(YOLOV5_PATH):
        print("Cloning YOLOv5 repository...")
        os.system("git clone https://github.com/ultralytics/yolov5.git")
        
    if YOLOV5_PATH not in sys.path:
        sys.path.append(YOLOV5_PATH)

def load_custom_yolov5m(weights_path, data_yaml_path):
    """
    Load YOLOv5m model with custom weights
    """
    try:
        setup_yolov5()
        from models.experimental import attempt_load
        
        # Load model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        model = attempt_load(weights_path, device=device)
        
        # Load class names
        with open(data_yaml_path, 'r') as f:
            data_yaml = yaml.safe_load(f)
            class_names = data_yaml['names']
        model.names = class_names
        
        # Set model parameters
        model.conf = 0.5  # Confidence threshold
        model.iou = 0.45  # NMS IoU threshold
        model.eval()
        
        print("Successfully loaded YOLOv5m with custom weights")
        return model, device
        
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def listen_for_command(recognizer, timeout=5):
    """
    Listen for voice command using Vosk with timeout
    Returns: Recognized text or None if no clear command
    """
    print("Listening for command...")
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, 
                   channels=1, 
                   rate=16000, 
                   input=True, 
                   frames_per_buffer=8192)
    stream.start_stream()

    try:
        start_time = cv2.getTickCount()
        audio_data = []
        
        while (cv2.getTickCount() - start_time) / cv2.getTickFrequency() < timeout:
            data = stream.read(4096, exception_on_overflow=False)
            audio_data.append(data)
            
            if recognizer.AcceptWaveform(data):
                result = recognizer.Result()
                result_dict = json.loads(result)
                
                # Check if we got a clear command
                if result_dict.get("text", "").strip():
                    return result_dict["text"].lower()
                
        return None
        
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

def process_frame(frame, model, device, conf_threshold=0.5):
    """
    Process a single frame through the YOLOv5 model
    """
    # Prepare frame for model
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(frame_rgb, (640, 640))
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    img = img.to(device).float() / 255.0

    # Inference
    with torch.no_grad():
        predictions = model(img)
        
    # Process predictions
    detected_objects = []
    
    # Get predictions from the first detection (index 0)
    pred = predictions[0].cpu().numpy()
    
    if len(pred) > 0:
        # Iterate through detections
        for detection in pred:
            # Get confidence scores
            scores = detection[4:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence >= conf_threshold:
                # Get box coordinates (center x, center y, width, height)
                x, y, w, h = detection[0:4]
                
                # Convert to corner coordinates
                img_height, img_width = frame.shape[:2]
                x1 = int((x - w/2) * img_width)
                y1 = int((y - h/2) * img_height)
                x2 = int((x + w/2) * img_width)
                y2 = int((y + h/2) * img_height)
                
                # Calculate center
                center_x = int(x * img_width)
                center_y = int(y * img_height)
                
                # Get class name
                class_name = model.names[class_id]
                
                # Store detection
                detected_objects.append({
                    'class': class_name,
                    'center': (center_x, center_y),
                    'confidence': float(conf),
                    'bbox': (x1, y1, x2, y2)
                })
                
                # Draw on frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
                
                # Add labels
                label = f"{class_name} {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                center_text = f"({center_x}, {center_y})"
                cv2.putText(frame, center_text, (center_x + 10, center_y - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    return frame, detected_objects

def main():
    # Initialize paths
    weights_path = "/Users/evansvetina/Downloads/best.pt"
    data_yaml_path = "/Users/evansvetina/Downloads/ISEF training dataset.v1i.yolov5pytorch (1)/data.yaml"
    vosk_model_path = "/Users/evansvetina/Downloads/vosk-model-small-en-us-0.15"

    # Check paths exist
    for path in [weights_path, data_yaml_path, vosk_model_path]:
        if not os.path.exists(path):
            print(f"Error: Path does not exist: {path}")
            sys.exit(1)

    # Initialize model
    print("Loading YOLOv5 model...")
    model, device = load_custom_yolov5m(weights_path, data_yaml_path)

    # Initialize text-to-speech
    print("Initializing text-to-speech...")
    engine = pyttsx3.init()

    # Initialize speech recognition
    print("Initializing speech recognition...")
    vosk_model = Model(vosk_model_path)
    recognizer = KaldiRecognizer(vosk_model, 16000)

    # Initialize camera
    print("Initializing camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        sys.exit(1)

    print("System ready! Press 'q' to quit.")
    
    try:
        while True:
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break

            # Process frame
            frame, detected_objects = process_frame(frame, model, device)

            # Show frame
            cv2.imshow('YOLOv5m Custom Object Detection', frame)

            # Listen for command (non-blocking)
            if cv2.waitKey(1) & 0xFF == ord('v'):  # Press 'v' to activate voice command
                command = listen_for_command(recognizer)
                
                if command:
                    print(f"Command recognized: {command}")
                    
                    # Look for objects mentioned in command
                    found = False
                    for obj in detected_objects:
                        if obj['class'].lower() in command:
                            print(f"Found {obj['class']} at center point {obj['center']}")
                            engine.say(f"The {obj['class']} is at coordinates {obj['center']}")
                            engine.runAndWait()
                            found = True
                            break
                    
                    if not found:
                        print("No matching object found for the command")
                        engine.say("Sorry, I couldn't find that object in the current frame")
                        engine.runAndWait()

            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()

    finally:
        print("Cleaning up...")
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
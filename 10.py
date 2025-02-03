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
import traceback
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
        
        # Load model with lower confidence threshold
        model = attempt_load(weights_path, device=device)
        model.conf = 0.1  # Much lower confidence threshold for testing
        model.iou = 0.45  # NMS IoU threshold
        
        # Load class names and print them
        with open(data_yaml_path, 'r') as f:
            data_yaml = yaml.safe_load(f)
            class_names = data_yaml['names']
            print("\nDataset configuration:")
            print(json.dumps(data_yaml, indent=2))
            
        model.names = class_names
        print(f"\nLoaded classes: {class_names}")
        print("\nModel configuration:")
        print(f"- Confidence threshold: {model.conf}")
        print(f"- IOU threshold: {model.iou}")
        
        model.eval()
        return model, device
        
    except Exception as e:
        print(f"Error loading model: {e}")
        traceback.print_exc()
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

def process_frame(frame, model, device, conf_threshold=0.3):
    """
    Process a single frame through the YOLOv5 model
    """
    try:
        # Preserve original image dimensions
        original_dims = frame.shape[:2]
        
        # Prepare frame for model (maintaining aspect ratio)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Pad image to square while maintaining aspect ratio
        height, width = frame_rgb.shape[:2]
        max_dim = max(height, width)
        pad_top = (max_dim - height) // 2
        pad_bottom = max_dim - height - pad_top
        pad_left = (max_dim - width) // 2
        pad_right = max_dim - width - pad_left
        
        frame_padded = cv2.copyMakeBorder(frame_rgb, pad_top, pad_bottom, pad_left, pad_right, 
                                        cv2.BORDER_CONSTANT, value=(0, 0, 0))
        
        # Resize to model input size
        img = cv2.resize(frame_padded, (640, 640))
        
        # Convert to tensor and normalize
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        img = img.to(device).float() / 255.0

        # Inference
        with torch.no_grad():
            predictions = model(img)
            if isinstance(predictions, (list, tuple)):
                predictions = predictions[0]
            
            # Convert to numpy and move to CPU
            pred = predictions.cpu().numpy()
        
        # Process predictions
        detected_objects = []
        num_detections = pred.shape[1]
        
        # Debug print
        print(f"Number of potential detections: {num_detections}")
        
        for i in range(num_detections):
            detection = pred[0, i, :]
            
            # Get coordinates and scores
            x, y, w, h = detection[0:4]
            obj_conf = detection[4]
            class_scores = detection[5:]
            
            # Get class with highest confidence
            class_id = np.argmax(class_scores)
            class_conf = class_scores[class_id]
            
            # Combined confidence
            confidence = float(obj_conf * class_conf)
            
            # Detailed debug prints for all detections
            print(f"\nDetection {i}:")
            print(f"- Class: {model.names[class_id]}")
            print(f"- Object confidence: {obj_conf:.3f}")
            print(f"- Class confidence: {class_conf:.3f}")
            print(f"- Combined confidence: {confidence:.3f}")
            print(f"- Coordinates (normalized): x={x:.3f}, y={y:.3f}, w={w:.3f}, h={h:.3f}")
            
            if confidence >= 0.01:  # Show even very low confidence detections
                # Convert normalized coordinates back to original image space
                x1 = int((x - w/2) * width)
                y1 = int((y - h/2) * height)
                x2 = int((x + w/2) * width)
                y2 = int((y + h/2) * height)
                center_x = int(x * width)
                center_y = int(y * height)
                
                # Get class name
                class_name = model.names[class_id]
                
                # Store detection
                detected_objects.append({
                    'class': class_name,
                    'center': (center_x, center_y),
                    'confidence': confidence,
                    'bbox': (x1, y1, x2, y2)
                })
                
                # Draw on frame with thicker lines for visibility
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
                
                # Add labels with better visibility
                label = f"{class_name} {confidence:.2f}"
                # Add background to text for better visibility
                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 255, 0), -1)
                cv2.putText(frame, label, (x1, y1 - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        return frame, detected_objects
        
    except Exception as e:
        print(f"Error in process_frame: {e}")
        traceback.print_exc()
        return frame, []

def main():
    try:
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

        print("System ready! Press 'q' to quit, 'v' for voice commands.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break

            frame, detected_objects = process_frame(frame, model, device)
            cv2.imshow('YOLOv5m Custom Object Detection', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('v'):
                command = listen_for_command(recognizer)
                if command:
                    print(f"Command recognized: {command}")
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

            if key == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nGracefully shutting down...")
    except Exception as e:
        print(f"Error during execution: {e}")
        traceback.print_exc()
    finally:
        print("Cleaning up...")
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
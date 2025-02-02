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
    YOLOV5_PATH = "yolov5"
    if not os.path.exists(YOLOV5_PATH):
        os.system("git clone https://github.com/ultralytics/yolov5.git")
    if YOLOV5_PATH not in sys.path:
        sys.path.append(YOLOV5_PATH)

def load_custom_yolov5m(weights_path, data_yaml_path):
    try:
        setup_yolov5()
        from models.experimental import attempt_load
        
        device = torch.device('cpu')
        model = attempt_load(weights_path, device=device)
        
        with open(data_yaml_path, 'r') as f:
            data_yaml = yaml.safe_load(f)
            class_names = data_yaml['names']
            
        model.names = class_names
        model.eval()
        
        print("Successfully loaded YOLOv5m with custom weights")
        return model, device
        
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def preprocess_image(img, device):
    img = cv2.resize(img, (640, 640))
    
    img = img[:, :, ::-1].transpose(2, 0, 1)
    
    img = np.ascontiguousarray(img) / 255.0
    
    mg = torch.from_numpy(img).float().unsqueeze(0).to(device)
    
    return img

print("Installing required packages...")

os.system("pip install -r yolov5/requirements.txt")

weights_path = "/Users/nikhil/Downloads/best.pt"

data_yaml_path = "/Users/nikhil/Downloads/ISEF training dataset.v1i.yolov5pytorch (1)/data.yaml"

try:
    model, device = load_custom_yolov5m(weights_path, data_yaml_path)
    
    model.conf = 0.25
    
    model.iou = 0.45
    
    print(f"Model loaded successfully on: {device}")
    
except Exception as e:
    print(f"Error initializing model: {e}")
    exit()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        img = preprocess_image(frame, device)
        
        with torch.no_grad():
            predictions = model(img)[0]
        
        detected_coordinates = []
        for det in predictions:
            if len(det) >= 6:
                x_min, y_min, x_max, y_max, conf, cls = det[:6].tolist()
                
                if conf >= model.conf:
                    center_x = (x_min + x_max) / 2
                    center_y = (y_min + y_max) / 2
                    detected_coordinates.append((int(center_x), int(center_y)))
        
        print("Detected coordinates:", detected_coordinates)
        
        additional_coordinates = [(0, 0), (0, 0)]  # Placeholder for user input
        print("Additional coordinates:", additional_coordinates)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"Error during execution: {e}")

finally:
    cap.release()
    cv2.destroyAllWindows()

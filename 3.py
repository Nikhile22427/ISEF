import cv2
import torch
import numpy as np
import pyttsx3
import os
import wave
from vosk import Model, KaldiRecognizer
import pyaudio


# Load the pre-trained YOLOv5 model from Ultralytics
model = torch.hub.load('ultralytics/yolov5', 'yolov5m')  # You can choose 'yolov5m', 'yolov5l', or 'yolov5x' based on your needs
device = torch.device('cpu')
# Initialize text-to-speech engine
engine = pyttsx3.init()

# Load Vosk model for speech recognition
model_path = "/Users/nikhil/Downloads/vosk-model-small-en-us-0.15"  # Update this path with your model directory
if not os.path.exists(model_path):
    print("Error: Vosk model path is invalid.")
    exit()

vosk_model = Model(model_path)
recognizer = KaldiRecognizer(vosk_model, 16000)

# Open the webcam (0 is the default camera index)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

def listen_for_command():
    print("Listening for command...")

    # Use Pyaudio for microphone recording
    p = pyaudio.PyAudio()

    # Open a stream for audio recording
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

    detected_objects = []
    # Loop over the bounding boxes
    for box in boxes:
        x_min, y_min, x_max, y_max, conf, class_id = box
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2

        detected_objects.append({
            'class': model.names[int(class_id)],
            'center': (int(center_x), int(center_y)),
            'confidence': conf  
        })

        # Draw the bounding box and center point
        cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)  # Green box
        cv2.circle(frame, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)  # Red circle at the center

       
        label = f"{model.names[int(class_id)]} {conf:.2f}"
        cv2.putText(frame, label, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        center_text = f"({int(center_x)}, {int(center_y)})"
        cv2.putText(frame, center_text, (int(center_x) + 10, int(center_y) - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)  # Yellow color for center coordinates

    # Show the frame with bounding boxes and centers
    cv2.imshow('YOLOv5 Object Detection', frame)

    command = listen_for_command()

    if command:
        print(f"Command recognized: {command}")
        # Process the command
        for obj in detected_objects:
            if obj['class'].lower() in command.lower():  # Convert both to lowercase
                print(f"Selected object: {obj['class']} at center point {obj['center']}")
                engine.say(f"The center of the {obj['class']} is at {obj['center']}")
                engine.runAndWait()
                break
        else:
            print("No matching object found for the command.")


    # Break the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture object and close the OpenCV window
cap.release()
cv2.destroyAllWindows()

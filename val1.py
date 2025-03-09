from ultralytics import YOLO

# Load a model
model = YOLO("/Users/nikhil/Downloads/best (1).pt")

# Validate with a custom dataset
metrics = model.val(data="/Users/nikhil/Downloads/ISEF training dataset.v1i.yolov5pytorch (1)/data.yaml")
print(metrics.box.map)  # map50-95
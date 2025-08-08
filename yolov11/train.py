from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("../yolo11l-cls.pt")
data_path="/mnt/workspace/data_3"
# Train the model on the COCO8 dataset for 100 epochs
train_results = model.train(
    data=data_path,  # Path to dataset configuration file
    epochs=100,  # Number of training epochs
    imgsz=224,  # Image size for training
    device="0",  # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
)
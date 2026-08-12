import os
import urllib.request
from ultralytics import YOLO

def download_depth_model():
    # MiDaS v2.1 Small model weights (Fast, suitable for real-time video)
    midas_url = "https://github.com/isl-org/MiDaS/releases/download/v2_1/midas_v21_small_256.pt"
    dest_path = "weights/midas_v21_small.pt"
    if not os.path.exists(dest_path):
        print(f"Downloading Depth Model (MiDaS) from {midas_url}...")
        urllib.request.urlretrieve(midas_url, dest_path)
        print("Depth model downloaded successfully.")
    else:
        print("Depth model already exists.")

def download_models():
    print("Downloading YOLOv8 and Depth models to the 'weights' directory...\n")
    
    # Model 1: Crowd/Pedestrian Tracking (Nano - ultra fast)
    print("--- 1. Downloading YOLOv8 Nano (Crowd/Pedestrians) ---")
    YOLO("weights/yolov8n.pt")
    
    # Model 2: Vehicles/Traffic (Small - slightly more accurate)
    print("\n--- 2. Downloading YOLOv8 Small (Vehicles/Traffic) ---")
    YOLO("weights/yolov8s.pt")

    # Model 3: Anomalies/Baggage/Weapons (Medium - better for small objects like backpacks)
    print("\n--- 3. Downloading YOLOv8 Medium (Anomalies/Baggage) ---")
    YOLO("weights/yolov8m.pt")

    # Model 4: Flow of Things (Segmentation / Tracking context)
    print("\n--- 4. Downloading YOLOv8 Nano Segmentation (Flow & Detailed Tracking) ---")
    YOLO("weights/yolov8n-seg.pt")

    # Depth Model: MiDaS
    print("\n--- 5. Downloading Depth Model ---")
    download_depth_model()

    print("\nAll 4 YOLO models and the Depth model successfully downloaded to weights/")

if __name__ == "__main__":
    if not os.path.exists("weights"):
        os.makedirs("weights")
    download_models()

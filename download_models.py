"""
Model Downloader — Downloads all AI model weights to the weights/ directory.

Models:
  1. YOLOv8 Nano      — Crowd/Pedestrian tracking
  2. YOLOv8 Small     — Vehicles/Traffic detection
  3. YOLOv8 Medium    — Anomaly/Baggage detection
  4. YOLOv8 Nano-Seg  — Flow & segmentation tracking
  5. MiDaS v2.1 Small — Monocular depth estimation
  6. SmolVLM2 2.2B Q4 — Vision Language Model (scene understanding)
"""

import os
import urllib.request
from ultralytics import YOLO

WEIGHTS_DIR = "weights"


def download_depth_model():
    """Download MiDaS v2.1 Small model weights."""
    midas_url = "https://github.com/isl-org/MiDaS/releases/download/v2_1/midas_v21_small_256.pt"
    dest_path = os.path.join(WEIGHTS_DIR, "midas_v21_small.pt")
    if not os.path.exists(dest_path):
        print(f"Downloading Depth Model (MiDaS) from {midas_url}...")
        urllib.request.urlretrieve(midas_url, dest_path)
        print("  ✓ Depth model downloaded.")
    else:
        print("  ✓ Depth model already exists.")


def download_vlm_model():
    """Download SmolVLM2 2.2B Instruct Q4_K_M GGUF + mmproj from HuggingFace."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("  ✗ huggingface-hub not installed. Skipping VLM download.")
        print("    Run: pip install huggingface-hub")
        return

    repo_id = "ggml-org/SmolVLM2-2.2B-Instruct-GGUF"

    # Main model weights
    model_dest = os.path.join(WEIGHTS_DIR, "SmolVLM2-2.2B-Instruct-Q4_K_M.gguf")
    if not os.path.exists(model_dest):
        print(f"Downloading SmolVLM2 2.2B Q4_K_M from {repo_id}...")
        hf_hub_download(
            repo_id=repo_id,
            filename="SmolVLM2-2.2B-Instruct-Q4_K_M.gguf",
            local_dir=WEIGHTS_DIR,
        )
        print("  ✓ VLM model downloaded.")
    else:
        print("  ✓ VLM model already exists.")

    # Multimodal projector (required for vision)
    mmproj_dest = os.path.join(WEIGHTS_DIR, "mmproj-SmolVLM2-2.2B-Instruct-f16.gguf")
    if not os.path.exists(mmproj_dest):
        print(f"Downloading SmolVLM2 mmproj from {repo_id}...")
        hf_hub_download(
            repo_id=repo_id,
            filename="mmproj-SmolVLM2-2.2B-Instruct-f16.gguf",
            local_dir=WEIGHTS_DIR,
        )
        print("  ✓ VLM mmproj downloaded.")
    else:
        print("  ✓ VLM mmproj already exists.")


def download_models():
    print("━━━ Downloading AI Models to 'weights/' ━━━\n")

    # Model 1: Crowd/Pedestrian (Nano — ultra fast)
    print("1. YOLOv8 Nano (Crowd/Pedestrians)")
    YOLO(os.path.join(WEIGHTS_DIR, "yolov8n.pt"))
    print("  ✓ Done.\n")

    # Model 2: Vehicles/Traffic (Small — slightly more accurate)
    print("2. YOLOv8 Small (Vehicles/Traffic)")
    YOLO(os.path.join(WEIGHTS_DIR, "yolov8s.pt"))
    print("  ✓ Done.\n")

    # Model 3: Anomalies/Baggage (Medium — better for small objects)
    print("3. YOLOv8 Medium (Anomalies/Baggage)")
    YOLO(os.path.join(WEIGHTS_DIR, "yolov8m.pt"))
    print("  ✓ Done.\n")

    # Model 4: Flow/Segmentation (Nano-Seg)
    print("4. YOLOv8 Nano Segmentation (Flow Tracking)")
    YOLO(os.path.join(WEIGHTS_DIR, "yolov8n-seg.pt"))
    print("  ✓ Done.\n")

    # Model 5: Depth
    print("5. MiDaS Depth Model")
    download_depth_model()
    print()

    # Model 6: VLM (SmolVLM2)
    print("6. SmolVLM2 2.2B Vision Language Model (Q4_K_M)")
    download_vlm_model()
    print()

    print("━━━ All models downloaded to weights/ ━━━")


if __name__ == "__main__":
    if not os.path.exists(WEIGHTS_DIR):
        os.makedirs(WEIGHTS_DIR)
    download_models()

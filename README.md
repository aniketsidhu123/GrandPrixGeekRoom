# Crowd Flow Optimiser

Real-time crowd simulation, AI-powered vision pipeline, bottleneck forecasting, and automated rerouting engine. A predictive digital twin for venue crowd management.

## Features

- **AI Vision Pipeline:** End-to-end video analysis using YOLOv8 (Crowd, Traffic, Anomalies, Flow Seg) and MiDaS (Depth Estimation).
- **Vision Language Model (VLM):** Integrated SmolVLM2 (2.2B) for high-level scene understanding and safety hazard descriptions.
- **Hardware Acceleration:** Native PyTorch CUDA 12.8 support and GPU-compiled `llama-cpp-python` for fast local inference.
- **Real-time Crowd Simulation:** Social Force Model physics mapping 2D camera coordinates to a 3D grid layout.
- **Density Heatmap & Analytics:** Fruin Level-of-Service classification and dynamic directional updates.
- **15-Minute Predictive Forecasting:** Automated bottleneck detection and proactive rerouting.

## Tech Stack

- **Backend:** Python, FastAPI, WebSockets, PyTorch, ultralytics (YOLO), llama-cpp-python
- **Frontend:** Vanilla HTML/CSS/JS with Canvas API
- **AI Models:** YOLOv8 (Nano/Small/Medium), MiDaS v2.1, SmolVLM2-2.2B-Instruct (GGUF)

## Getting Started

### Prerequisites

- **Windows OS**
- **Python 3.10+**
- **NVIDIA GPU** (Recommended: 8GB+ VRAM, e.g., RTX 4060)
- **CUDA Toolkit** and **C++ Build Tools** (Required for GPU acceleration of the VLM)

### Installation & Setup

We provide automated batch scripts for Windows to handle virtual environments, dependencies, and downloading all AI model weights (~1.2GB).

1. Clone the repository:
   ```cmd
   git clone https://github.com/aniketsidhu123/GrandPrixGeekRoom.git
   cd GrandPrixGeekRoom
   ```

2. Run the automated setup script. This will create `.venv`, install PyTorch with CUDA, compile `llama-cpp-python` for GPU, and download all models to the `weights/` directory:
   ```cmd
   .\setup.bat
   ```

### Running the Server

Once setup is complete, launch the backend and frontend simultaneously:

```cmd
.\start.bat
```

The terminal will display the Uvicorn server logs, and it will automatically open the dashboard in your default browser at `http://localhost:8001/static/index.html`.

## Usage

### 1. AI Vision Pipeline
- Click **Add Files** or **Add Folder** to select local video files (`.mp4`, `.avi`, etc.)
- Click **Scan** to process the videos through the AI pipeline.
- The dashboard will display live FPS, Latency, VRAM usage, and active models.
- The VLM will periodically generate text-based scene intelligence reports.

### 2. Simulation & Digital Twin
- Click **Spawn** to add synthetic agents to the simulation manually.
- Use the **speed controls** (1×, 2×, 5×, 10×) to adjust simulation time.
- Toggle visualization layers (Heatmap, Flow, Paths, Agents, Labels).
- Monitor density metrics, alerts, and 15-minute predictions in the sidebars.

# Crowd Guard

Real-time crowd simulation, AI-powered vision pipeline, bottleneck forecasting, and automated rerouting engine. A predictive digital twin for venue crowd management.

## Features

- **AI Vision Pipeline:** End-to-end video analysis using YOLOv8 (Crowd, Traffic, Anomalies, Flow Seg) and MiDaS (Depth Estimation).
- **Vision Language Model (VLM):** Integrated SmolVLM2 (2.2B) for high-level scene understanding and safety hazard descriptions.
- **Hardware Acceleration:** Native PyTorch CUDA 12.8 support and GPU-compiled `llama-cpp-python` for fast local inference.
- **Real-time Crowd Simulation:** Social Force Model physics mapping 2D camera coordinates to a 3D grid layout.
- **Density Heatmap & Analytics:** Fruin Level-of-Service classification and dynamic directional updates.
- **15-Minute Predictive Forecasting:** Automated bottleneck detection with density-weighted A* rerouting.
- **Autopilot:** Diverts gate arrivals on its own when an operator doesn't respond in time, with a full activity log and configurable response window.

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

> **macOS / Linux:** use `./setup.sh` and `./start.sh` instead of the `.bat` scripts below. There's no CUDA support on these platforms — PyTorch runs on CPU (or MPS on Apple Silicon), and the VLM's `llama-cpp-python` build is optional (Metal-accelerated on macOS, CPU-only otherwise). See the comments inside `setup.sh` for the exact commands.

### Installation & Setup

We provide automated batch scripts for Windows to handle virtual environments, dependencies, and downloading all AI model weights (~1.2GB).

1. Clone the repository:
   ```cmd
   git clone https://github.com/ishitathakur22/crowd-guard.git
   cd crowd-guard
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

No NVIDIA GPU? Use `.\start-cpu.bat` instead — everything runs on CPU.

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
- Adjust **Expected crowd size**, **Doors open**, and **Surge intensity** in Event Setup, or pick a **Day type** preset (Weekday commute / Match day / Sell-out surge), to change how heavily the venue fills up.
- Monitor density metrics, alerts, and 15-minute predictions in the sidebars.

### 3. Recommended Actions & Autopilot
- The **Recommended Actions** panel lists zones the 15-minute forecast expects to hit critical density, soonest first, each with a one-click **Divert** button that reroutes a share of a congested gate's new arrivals to a less-loaded one.
- Turn on **Autopilot** to have the system apply that same divert automatically if nobody responds within a configurable window (Now / 15s / 30s / 60s / 90s / 2m). It only acts on pile-ups that are both imminent and already busy right now, never closes or throttles a gate, and everything it does is written to the **Activity Log** for review.

## Project Structure

```
crowd-guard/
├── backend/          # FastAPI app, simulation engine, physics, prediction, autopilot
├── frontend/         # Dashboard UI (HTML/CSS/JS, Canvas rendering)
├── scripts/          # One-off data-prep scripts (e.g. arrival profile builder)
└── weights/          # Downloaded AI model weights (populated by download_models.py)
```

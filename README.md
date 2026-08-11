# Crowd Flow Optimiser

Real-time crowd simulation, bottleneck forecasting, and automated rerouting engine. A predictive digital twin for venue crowd management.

## Features

- **Real-time crowd simulation** with Social Force Model physics
- **Density heatmap** with Fruin Level-of-Service classification
- **Bottleneck detection** and automated rerouting
- **15-minute predictive forecasting**
- **Gate control** with throttling and closure
- **Digital signage** with dynamic directional updates
- **Live alert feed** with severity levels
- **Multi-layer canvas rendering** (heatmap, flow vectors, paths, agents)

## Tech Stack

- **Backend:** Python, FastAPI, WebSockets, NumPy
- **Frontend:** Vanilla HTML/CSS/JS with Canvas API
- **Simulation:** A* pathfinding, Social Force Model, density analysis

## Getting Started

### Prerequisites

- Python 3.10+

### Installation

```bash
# Clone the repository
git clone https://github.com/aniketsidhu123/GrandPrixGeekRoom.git
cd GrandPrixGeekRoom

# Create virtual environment
python -m venv .venv

# Activate it
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source backend/venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000/static/index.html** in your browser.

## Usage

1. Click **Spawn** to add agents to the simulation
2. Use the **speed controls** (1×, 2×, 5×, 10×) to adjust simulation speed
3. Toggle **visualization layers** (Heatmap, Flow, Paths, Agents, Labels)
4. Monitor **density metrics**, **alerts**, and **predictions** in the sidebars
5. Control **gates** to throttle or close entry points

## API Docs

Interactive API documentation is available at **http://localhost:8000/docs** when the server is running.

## License

MIT

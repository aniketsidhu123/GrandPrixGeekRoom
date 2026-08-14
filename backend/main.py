from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import List, Optional
import asyncio
import os
import json

from .simulation import SimulationEngine, TICK_DURATION_SEC
from .vision.pipeline import VisionPipeline

app = FastAPI(
    title="Crowd Flow Optimiser",
    description="Real-time crowd simulation, bottleneck forecasting, and automated rerouting engine.",
    version="2.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure frontend directory exists
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'frontend')
if not os.path.exists(frontend_dir):
    os.makedirs(frontend_dir)

app.mount("/static", StaticFiles(directory=frontend_dir, html=True), name="static")

sim_engine = SimulationEngine()

# Initialize Vision Pipeline (but don't start capture yet)
vision_pipeline = VisionPipeline(grid_width=sim_engine.width, grid_height=sim_engine.height)
vision_pipeline.on_agents_detected = sim_engine.ingest_real_data

# Vision update state (pushed to frontend via WebSocket)
latest_vision_update = {}

def _handle_vision_update(data: dict):
    global latest_vision_update
    latest_vision_update = data

vision_pipeline.on_vision_update = _handle_vision_update

@app.get("/")
async def root():
    return RedirectResponse(url="/static/index.html")

@app.get("/health")
async def health():
    return {"status": "ok"}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SIMULATION LOOP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(simulation_loop())

async def simulation_loop():
    sim_engine.running = True
    loop = asyncio.get_event_loop()
    while sim_engine.running:
        if sim_engine.sim_speed > 0:
            # The physics substeps and the 15-minute forecast are CPU-bound.
            # Running them in a worker thread keeps the WebSocket broadcasts
            # flowing instead of stalling every time a forecast fires.
            await loop.run_in_executor(None, sim_engine.update)
            # Each tick represents TICK_DURATION_SEC of simulated time, so the
            # wall-clock gap between ticks is what the speed multiplier scales.
            await asyncio.sleep(max(0.02, TICK_DURATION_SEC / sim_engine.sim_speed))
        else:
            await asyncio.sleep(0.5)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# WEBSOCKET CONNECTION MANAGER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        # Send initial grid config + venue metadata
        await websocket.send_json({
            "type": "grid_config",
            "width": sim_engine.width,
            "height": sim_engine.height,
            "grid": sim_engine.grid,
            "gates": [g.model_dump() for g in sim_engine.gate_objects],
            "exits": [{"x": e.x, "y": e.y} for e in sim_engine.exits],
            "concessions": [{"x": c.x, "y": c.y} for c in sim_engine.concessions],
            "signs": [s.model_dump() for s in sim_engine.signs],
        })

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.append(connection)
        for conn in disconnected:
            if conn in self.active_connections:
                self.active_connections.remove(conn)

manager = ConnectionManager()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# WEBSOCKET ENDPOINT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            try:
                command = json.loads(data)
                action = command.get("action")
                
                if action == "spawn":
                    count = command.get("count", 10)
                    sim_engine.spawn_agents(count)
                    
                elif action == "clear":
                    sim_engine.agents = []
                    sim_engine.alerts = []
                    sim_engine.bottlenecks = []
                    sim_engine.predictions = []
                    sim_engine.suggested_routes.clear()
                    
                elif action == "set_speed":
                    speed = command.get("speed", 1.0)
                    sim_engine.set_sim_speed(speed)
                    
                elif action == "gate_control":
                    gate_id = command.get("gate_id")
                    gate_action = command.get("gate_action", "OPEN_FULL")
                    rate = command.get("rate", 3.5)
                    if gate_id:
                        sim_engine.actuate_gate(gate_id, gate_action, rate)
                        
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STATE BROADCASTER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def broadcast_state():
    while True:
        if manager.active_connections:
            state = sim_engine.get_state()
            heatmap_data = sim_engine.get_heatmap_data()
            
            await manager.broadcast({
                "type": "state_update",
                "state": state,
                "heatmap": heatmap_data,
            })
        sleep_time = max(0.1, 0.5 / sim_engine.sim_speed) if sim_engine.sim_speed > 0 else 0.5
        await asyncio.sleep(sleep_time)

async def broadcast_vision_state():
    """Broadcast vision pipeline metrics to all connected clients."""
    while True:
        if manager.active_connections and latest_vision_update:
            await manager.broadcast({
                "type": "vision_update",
                "data": latest_vision_update,
            })
        await asyncio.sleep(0.5)

@app.on_event("startup")
async def start_broadcaster():
    asyncio.create_task(broadcast_state())
    asyncio.create_task(broadcast_vision_state())

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# REST API ENDPOINTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class VisionBoundingBox(BaseModel):
    track_id: int
    x: float
    y: float
    confidence: float

class VisionIngestionPayload(BaseModel):
    camera_id: str
    timestamp: float
    detections: List[VisionBoundingBox]

@app.post("/ingest/vision")
async def ingest_vision_data(payload: VisionIngestionPayload):
    """
    Mock endpoint for the YOLOv8/DeepSORT vision pipeline.
    In a real scenario, this would map 2D camera coordinates to the 3D grid layout.
    """
    print(f"Received {len(payload.detections)} detections from {payload.camera_id}")
    return {"status": "success", "processed": len(payload.detections)}

@app.post("/api/vision/start")
async def start_vision_pipeline():
    """Start the real-time AI vision pipeline (webcam)."""
    if not vision_pipeline.running:
        vision_pipeline.set_source(0)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, vision_pipeline.initialize)
        if not result["loaded"]:
            return {"status": "error", "detail": "No vision models could be loaded", **result}
        asyncio.create_task(vision_pipeline.start())
        return {"status": "started", "mode": "live", **result}
    return {"status": "already_running", "mode": "live"}

@app.post("/api/vision/stop")
async def stop_vision_pipeline():
    """Stop the real-time AI vision pipeline."""
    vision_pipeline.stop()
    return {"status": "stopped"}

class VideoScanRequest(BaseModel):
    paths: List[str]  # List of file paths and/or folder paths

@app.post("/api/vision/scan-video")
async def scan_video(request: VideoScanRequest):
    """
    Scan one or more video files, or entire folders of videos.
    Pass file paths and/or folder paths. Folders will be scanned recursively
    for .mp4, .avi, .mkv, .mov, .wmv, .flv, .webm files.
    """
    if vision_pipeline.running:
        vision_pipeline.stop()
    # Run blocking model loading in a thread
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, vision_pipeline.initialize)
    asyncio.create_task(vision_pipeline.scan_videos(request.paths))
    return {"status": "scanning", "paths": request.paths}

@app.get("/api/vision/status")
async def vision_status():
    """Return current vision pipeline status."""
    from .vision.model_manager import model_manager
    return {
        "running": vision_pipeline.running,
        "mode": vision_pipeline.mode,
        "fps": vision_pipeline.fps,
        "frame": vision_pipeline.frame_count,
        "total_frames": vision_pipeline.total_frames,
        "vlm_analysis": vision_pipeline.last_vlm_analysis,
        "model_manager": model_manager.get_status(),
    }

@app.get("/api/vision/pick-files")
async def pick_files():
    """Open a native file picker dialog and return selected video file paths."""
    import threading
    result = {"paths": []}

    def _pick():
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            files = filedialog.askopenfilenames(
                title="Select Video Files",
                filetypes=[
                    ("Video files", "*.mp4 *.avi *.mkv *.mov *.wmv *.flv *.webm"),
                    ("All files", "*.*"),
                ],
            )
            result["paths"] = list(files)
            root.destroy()
        except Exception as e:
            print(f"File picker error: {e}")

    # Run tkinter in a thread (it needs its own thread on Windows)
    t = threading.Thread(target=_pick)
    t.start()
    t.join(timeout=120)  # 2 min timeout
    return result

@app.get("/api/vision/pick-folder")
async def pick_folder():
    """Open a native folder picker dialog and return the selected folder path."""
    import threading
    result = {"path": ""}

    def _pick():
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            folder = filedialog.askdirectory(title="Select Video Folder")
            result["path"] = folder
            root.destroy()
        except Exception as e:
            print(f"Folder picker error: {e}")

    t = threading.Thread(target=_pick)
    t.start()
    t.join(timeout=120)
    return result

@app.get("/prediction")
async def get_prediction():
    """Return current forecast data."""
    return {
        "predictions": [p.model_dump() for p in sim_engine.predictions],
        "horizon_minutes": sim_engine.prediction_engine.horizon_minutes,
    }

@app.get("/stats")
async def get_stats():
    """Return current simulation statistics."""
    return {
        "total_agents": len(sim_engine.agents),
        "total_rerouted": sum(1 for a in sim_engine.agents if a.rerouted),
        "bottleneck_count": len(sim_engine.bottlenecks),
        "hazard_count": len(sim_engine.hazard_zones),
        "sim_time": sim_engine.sim_time,
        "sim_speed": sim_engine.sim_speed,
        "alert_count": len(sim_engine.alerts),
        "clock_hour": round(sim_engine.arrival_model.hour_at(sim_engine.sim_time), 2),
        "ingress_per_hour": round(
            sim_engine.arrival_model.rate_at(sim_engine.sim_time) * 3600, 1
        ) if sim_engine.auto_ingress else 0.0,
    }

class GateControlRequest(BaseModel):
    action: str  # OPEN_FULL, THROTTLE_FLOW, CLOSE
    rate: float = 3.5

@app.post("/gate/{gate_id}/actuate")
async def actuate_gate(gate_id: str, request: GateControlRequest):
    """Manually control a gate."""
    sim_engine.actuate_gate(gate_id, request.action, request.rate)
    return {"status": "success", "gate_id": gate_id, "action": request.action}

class IngressConfigRequest(BaseModel):
    """All fields optional — only what is supplied gets changed."""
    enabled: Optional[bool] = None
    archetype: Optional[str] = None      # "gate" | "concourse" | "food"
    attendance: Optional[int] = None     # people across the full 24h curve
    start_hour: Optional[float] = None   # wall-clock hour the sim starts at
    weekend: Optional[bool] = None
    scale: Optional[float] = None        # intensity multiplier for tuning


@app.get("/api/ingress")
async def get_ingress():
    """Current arrival model, including which sensor profile it came from."""
    model = sim_engine.arrival_model
    hour = model.hour_at(sim_engine.sim_time)
    return {
        "enabled": sim_engine.auto_ingress,
        "config": model.describe(),
        "clock_hour": round(hour, 2),
        "current_rate_per_sec": round(model.rate_at(sim_engine.sim_time), 3),
        "current_rate_per_hour": round(model.rate_at(sim_engine.sim_time) * 3600, 1),
        "curve": model.curve,
    }


@app.post("/api/ingress")
async def set_ingress(request: IngressConfigRequest):
    """Reconfigure the arrival model. Rebuilds it when the profile changes."""
    from .ingress import ArrivalModel

    if request.enabled is not None:
        sim_engine.auto_ingress = request.enabled

    model = sim_engine.arrival_model
    needs_rebuild = any(
        v is not None for v in (request.archetype, request.weekend)
    )

    if needs_rebuild:
        model = ArrivalModel(
            archetype=request.archetype or model.archetype,
            attendance=request.attendance or model.attendance,
            start_hour=model.start_hour if request.start_hour is None else request.start_hour,
            weekend=model.weekend if request.weekend is None else request.weekend,
            scale=model.scale if request.scale is None else request.scale,
        )
        sim_engine.arrival_model = model
        sim_engine.prediction_engine.set_arrival_model(model, sim_engine.gates)
    else:
        if request.attendance is not None:
            model.attendance = request.attendance
        if request.start_hour is not None:
            model.start_hour = request.start_hour
        if request.scale is not None:
            model.scale = request.scale

    return {"status": "success", "enabled": sim_engine.auto_ingress, **model.describe()}


class SpeedControlRequest(BaseModel):
    speed: float

@app.post("/simulation/speed")
async def set_speed(request: SpeedControlRequest):
    """Set simulation speed multiplier."""
    sim_engine.set_sim_speed(request.speed)
    return {"status": "success", "speed": sim_engine.sim_speed}

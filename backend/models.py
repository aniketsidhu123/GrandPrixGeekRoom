from pydantic import BaseModel, Field
from typing import List, Tuple, Dict, Optional
from enum import Enum


class CellType(str, Enum):
    EMPTY = "empty"
    WALL = "wall"
    GATE = "gate"
    EXIT = "exit"
    CONCESSION = "concession"
    STAIRCASE = "staircase"
    CORRIDOR = "corridor"


class FruinLoS(str, Enum):
    """Fruin Level-of-Service classification based on spatial density (people/m²)."""
    A = "A"      # ρ < 0.31  — Free movement
    B = "B"      # 0.31–0.43 — Minor restrictions
    C = "C"      # 0.43–1.08 — Restricted gait
    D = "D"      # 1.08–1.54 — Severely restricted
    E = "E"      # 1.54–2.17 — Compressed, shuffling
    F = "F"      # 2.17–3.50 — Severe congestion, body contact
    CRITICAL = "CRITICAL"  # > 3.50 — Crush risk, shockwave danger


class AlertSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"


class GateAction(str, Enum):
    OPEN_FULL = "OPEN_FULL"
    THROTTLE_FLOW = "THROTTLE_FLOW"
    CLOSE = "CLOSE"


class Point(BaseModel):
    x: float
    y: float
    
    def __hash__(self):
        return hash((self.x, self.y))
        
    def __eq__(self, other):
        if not isinstance(other, Point):
            return False
        return self.x == other.x and self.y == other.y


class AgentProfile(BaseModel):
    """Demographic variance for Social Force Model auto-tuning."""
    label: str = "adult"
    desired_speed: float = 1.3     # m/s
    radius: float = 0.3            # m (body half-width)
    mass: float = 80.0             # kg
    A: float = 2000.0              # Repulsion strength
    B: float = 0.08                # Repulsion range
    stair_factor_up: float = 0.5   # Speed multiplier going up stairs
    stair_factor_down: float = 0.7 # Speed multiplier going down stairs


# Preset demographic profiles
AGENT_PROFILES = {
    "adult": AgentProfile(label="adult", desired_speed=1.3, radius=0.3, mass=80.0),
    "elderly": AgentProfile(label="elderly", desired_speed=0.8, radius=0.35, mass=70.0, A=2500.0, B=0.10),
    "child": AgentProfile(label="child", desired_speed=1.0, radius=0.2, mass=35.0, A=1500.0),
    "luggage": AgentProfile(label="luggage", desired_speed=0.9, radius=0.45, mass=100.0, A=2200.0, B=0.12),
}


class Agent(BaseModel):
    id: int
    pos: Point
    vel: Point = Point(x=0.0, y=0.0)
    goal: Point
    path: List[Point] = []
    status: str = "moving"
    color: str = "#3b82f6"
    radius: float = 0.3       # Roughly 30cm radius for a person
    mass: float = 80.0        # 80 kg
    desired_speed: float = 1.3 # 1.3 m/s
    profile: str = "adult"
    rerouted: bool = False


class GateState(BaseModel):
    gate_id: str
    position: Point
    action: GateAction = GateAction.OPEN_FULL
    throughput_per_sec: float = 3.5
    target_rate_per_sec: float = 3.5
    status: str = "OPEN"
    agents_passed: int = 0


class ZoneDensity(BaseModel):
    """Per-zone density and Fruin LoS classification."""
    zone_id: str
    cell_x: int
    cell_y: int
    density: float          # people/m²
    agent_count: int
    area: float = 1.0       # m² (for grid cells, this is 1.0)
    los_level: FruinLoS
    flow_rate: float = 0.0  # agents/m/min
    trend: str = "stable"   # rising, falling, stable


class HazardZone(BaseModel):
    """Detected or predicted hazard zone."""
    zone_id: str
    cell_x: int
    cell_y: int
    current_density: float
    predicted_density: float
    severity: AlertSeverity
    fruin_los: FruinLoS
    time_to_choke_sec: float = 0.0


class RerouteRecommendation(BaseModel):
    """Auto-generated reroute suggestion."""
    reroute_id: str
    origin_zone: str
    congested_zone: str
    target_zone: str
    path_polyline: List[Point]
    signage_action: str = ""
    gate_actuations: List[Dict] = []


class SignageState(BaseModel):
    """Virtual digital signage board state."""
    sign_id: str
    position: Point
    message: str = ""
    direction: str = ""         # "LEFT", "RIGHT", "UP", "DOWN", "STRAIGHT"
    target_zone: str = ""
    active: bool = False


class AlertEvent(BaseModel):
    """System alert for the operations dashboard."""
    alert_id: str
    timestamp: float
    severity: AlertSeverity
    zone_id: str
    message: str
    density: float = 0.0
    los_level: FruinLoS = FruinLoS.A
    acknowledged: bool = False


class PredictionSnapshot(BaseModel):
    """A single timestep in the predictive forecast."""
    timestamp_offset_sec: float  # seconds into the future
    hazard_zones: List[HazardZone] = []
    reroutes: List[RerouteRecommendation] = []
    zone_densities: List[ZoneDensity] = []
    max_density: float = 0.0
    critical_zone_count: int = 0


class GridState(BaseModel):
    width: int
    height: int
    cells: List[List[str]]
    

class SimulationState(BaseModel):
    agents: List[Agent]
    real_agents: List[Dict] = []
    bottlenecks: List[Point]
    suggested_routes: Dict[int, List[Point]]
    zone_densities: List[ZoneDensity] = []
    hazard_zones: List[HazardZone] = []
    alerts: List[AlertEvent] = []
    gates: List[GateState] = []
    signs: List[SignageState] = []
    predictions: List[PredictionSnapshot] = []
    total_agents: int = 0
    total_rerouted: int = 0
    avg_density: float = 0.0
    max_density: float = 0.0
    sim_time_sec: float = 0.0
    sim_speed: float = 1.0

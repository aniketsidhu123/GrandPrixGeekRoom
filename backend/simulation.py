"""
Simulation Engine — Core crowd simulation with integrated density analysis,
prediction forecasting, dynamic rerouting, and gate actuation.
"""

import random
import time
import numpy as np
from typing import List, Dict, Tuple
from .models import (
    Point, Agent, CellType, SimulationState, FruinLoS,
    ZoneDensity, HazardZone, AlertEvent, GateState, GateAction,
    SignageState, PredictionSnapshot, RerouteRecommendation,
    AGENT_PROFILES, AlertSeverity
)
from .pathfinding import a_star_search, compute_alternative_routes
from .physics import SocialForceModel
from .density import DensityAnalyzer
from .prediction import PredictionEngine
from .ingress import ArrivalModel


# Simulated seconds advanced by one engine tick. The physics integrator runs at
# a finer dt, so a tick executes several substeps to cover this span — keeping
# the displayed clock honest about how much movement actually happened.
TICK_DURATION_SEC = 0.5


class SimulationEngine:
    def __init__(self, width: int = 40, height: int = 30):
        self.width = width
        self.height = height
        self.grid = self._generate_default_grid()
        self.agents: List[Agent] = []
        self.real_agents: List[Dict] = []  # Added for Vision Pipeline
        self.bottlenecks: List[Point] = []
        self.suggested_routes: Dict[int, List[Point]] = {}
        self.agent_id_counter = 0
        
        self.running = False
        self.sim_time = 0.0        # Simulation clock (seconds)
        self.sim_speed = 1.0       # Speed multiplier
        self.tick_count = 0
        
        # Physics engine. One tick covers TICK_DURATION_SEC of simulated time,
        # which takes several integrator substeps at dt=0.1s.
        self.sfm = SocialForceModel(dt=0.1)
        self.physics_substeps = max(1, int(round(TICK_DURATION_SEC / self.sfm.dt)))
        self.walls_cache = self._get_walls()
        
        # Density analyzer
        self.density_analyzer = DensityAnalyzer(width, height)
        self.zone_densities: List[ZoneDensity] = []
        self.hazard_zones: List[HazardZone] = []
        self.density_heatmap: List[List[float]] = []
        self.flow_vectors: List[Dict] = []
        
        # Prediction engine
        self.prediction_engine = PredictionEngine(width, height)
        self.predictions: List[PredictionSnapshot] = []
        self.prediction_interval = 10  # Run forecast every N ticks
        
        # Alert system
        self.alerts: List[AlertEvent] = []
        self.alert_history: List[AlertEvent] = []
        self.max_alerts = 50
        
        # Gates
        self.gate_objects: List[GateState] = []
        self.gates: List[Point] = [
            Point(x=0.0, y=10.0), Point(x=0.0, y=11.0), Point(x=0.0, y=12.0),
            Point(x=0.0, y=15.0), Point(x=0.0, y=16.0),
        ]
        self.exits: List[Point] = [
            Point(x=39.0, y=10.0), Point(x=39.0, y=11.0), 
            Point(x=39.0, y=20.0), Point(x=39.0, y=25.0),
        ]
        self.concessions: List[Point] = [
            Point(x=20.0, y=5.0), Point(x=20.0, y=25.0)
        ]
        
        # Digital signage
        self.signs: List[SignageState] = [
            SignageState(sign_id="sign_corridor_a", position=Point(x=10.0, y=14.0), message="→ Main Concourse", direction="RIGHT"),
            SignageState(sign_id="sign_fork_b", position=Point(x=22.0, y=10.0), message="→ Exit A", direction="RIGHT"),
            SignageState(sign_id="sign_fork_c", position=Point(x=22.0, y=20.0), message="↓ Exit B", direction="DOWN"),
        ]
        
        # Initialize gates on grid
        self._setup_venue_elements()

        # Arrival model — real hourly ingress curve measured from pedestrian
        # sensors, used both to drive live spawning and to tell the forecaster
        # how many people are still on their way in.
        self.arrival_model = ArrivalModel(
            archetype="gate",
            attendance=30_000,
            start_hour=16.0,   # simulation opens at 16:00, pre-event build-up
        )
        self.auto_ingress = True
        self.prediction_engine.set_arrival_model(self.arrival_model, self.gates)

        # Active reroute tracking
        self.active_reroutes: List[RerouteRecommendation] = []
        
    def _generate_default_grid(self) -> List[List[str]]:
        grid = [[CellType.EMPTY.value for _ in range(self.width)] for _ in range(self.height)]
        
        # Wall layout: creates a venue with corridors, chokepoints, and multiple paths
        # Central wall with gap (main chokepoint)
        for y in range(6, 24):
            if y not in (13, 14, 15, 16):  # Gap in the middle
                grid[y][15] = CellType.WALL.value
        
        # Upper barrier (forces traffic through specific corridors)
        for y in range(0, 8):
            grid[y][25] = CellType.WALL.value
            
        # Lower barrier
        for y in range(22, 30):
            grid[y][25] = CellType.WALL.value
            
        # Small obstacles creating secondary chokepoints
        for x in range(30, 34):
            grid[14][x] = CellType.WALL.value
            grid[15][x] = CellType.WALL.value
        
        # Corridor walls (upper passage)
        for x in range(8, 14):
            grid[5][x] = CellType.WALL.value
            grid[8][x] = CellType.WALL.value
        
        return grid
        
    def _get_walls(self) -> List[Tuple[int, int]]:
        walls = []
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y][x] == CellType.WALL.value:
                    walls.append((x, y))
        return walls
    
    def _setup_venue_elements(self):
        """Place gates, exits, concessions on the grid."""
        for i, g in enumerate(self.gates):
            gy, gx = int(g.y), int(g.x)
            if 0 <= gy < self.height and 0 <= gx < self.width:
                self.grid[gy][gx] = CellType.GATE.value
            self.gate_objects.append(GateState(
                gate_id=f"gate_{i}",
                position=g,
                throughput_per_sec=3.5,
                target_rate_per_sec=3.5,
            ))
            
        for e in self.exits:
            ey, ex = int(e.y), int(e.x)
            if 0 <= ey < self.height and 0 <= ex < self.width:
                self.grid[ey][ex] = CellType.EXIT.value
                
        for c in self.concessions:
            cy, cx = int(c.y), int(c.x)
            if 0 <= cy < self.height and 0 <= cx < self.width:
                self.grid[cy][cx] = CellType.CONCESSION.value

    def spawn_agents(self, count: int):
        """Spawn agents at gate positions with demographic variance."""
        profiles = list(AGENT_PROFILES.keys())
        profile_weights = [0.6, 0.15, 0.1, 0.15]  # adult, elderly, child, luggage
        
        spawned = 0
        for _ in range(count):
            # Check gate throttling
            gate_idx = random.randint(0, len(self.gates) - 1)
            gate_state = self.gate_objects[gate_idx]
            
            if gate_state.action == GateAction.CLOSE:
                continue  # Gate is closed
            
            start = self.gates[gate_idx]
            
            # Jitter to prevent stacking
            jitter_x = random.uniform(0.1, 0.9)
            jitter_y = random.uniform(-0.4, 0.4)
            actual_start = Point(x=start.x + jitter_x, y=start.y + jitter_y)
            
            # Choose destination
            if random.random() > 0.6:
                goal = random.choice(self.concessions)
            else:
                goal = random.choice(self.exits)
            
            # Choose demographic profile
            profile_name = random.choices(profiles, weights=profile_weights, k=1)[0]
            profile = AGENT_PROFILES[profile_name]
                
            agent = Agent(
                id=self.agent_id_counter,
                pos=actual_start,
                goal=goal,
                path=[],
                desired_speed=profile.desired_speed + random.uniform(-0.15, 0.15),
                radius=profile.radius,
                mass=profile.mass,
                profile=profile_name,
            )
            self.agent_id_counter += 1
            
            # Initial path calculation
            start_int = Point(x=float(int(actual_start.x)), y=float(int(actual_start.y)))
            goal_int = Point(x=float(int(goal.x)), y=float(int(goal.y)))
            
            path = a_star_search(self.grid, start_int, goal_int)
            if path:
                agent.path = path[1:] if len(path) > 1 else path
            
            self.agents.append(agent)
            spawned += 1

        # Record actual admissions (may be below `count` when gates are closed)
        # so the fallback ingress estimate reflects reality.
        self.prediction_engine.update_ingress_estimate(spawned)
        return spawned


    def ingest_real_data(self, real_agents_data: List[Dict]):
        """
        Phase 1: Purely Real Update.
        Accepts data from the Vision Pipeline and updates the real_agents list.
        Data format: [{"id": str, "x": float, "y": float, "type": str, "class_id": int, "conf": float, "depth": float}]
        """
        self.real_agents = real_agents_data
        
        # Note for Phase 2 (Hybrid Merge):
        # Real agents now influence synthetic agents via the Social Force Model.

        self.prediction_engine.update_ingress_estimate(len(real_agents_data))
            
    def update(self):
        """Main simulation tick."""
        self.tick_count += 1
        # The clock advances by the world time this tick actually simulates.
        # sim_speed controls how often ticks run in wall-clock time, not how
        # much simulated time each one represents.
        self.sim_time += TICK_DURATION_SEC

        # 0. Ingress — admit arrivals from the measured hourly curve
        if self.auto_ingress:
            arrivals = self.arrival_model.draw_arrivals(self.sim_time, TICK_DURATION_SEC)
            if arrivals:
                self.spawn_agents(arrivals)

        # 1. Update positions using Social Force Model
        for _ in range(self.physics_substeps):
            self.sfm.update_positions(self.agents, self.walls_cache, self.real_agents)
        
        # 2. Check path progress and goals
        for agent in self.agents:
            if agent.status == "arrived":
                continue
                
            # Reached next path node?
            if agent.path:
                next_node = agent.path[0]
                dist_sq = (agent.pos.x - (next_node.x + 0.5))**2 + (agent.pos.y - (next_node.y + 0.5))**2
                if dist_sq < 0.5:
                    agent.path.pop(0)
                    
            # Reached ultimate goal?
            dist_to_goal = (agent.pos.x - (agent.goal.x + 0.5))**2 + (agent.pos.y - (agent.goal.y + 0.5))**2
            if dist_to_goal < 1.0:
                gx, gy = int(agent.goal.x), int(agent.goal.y)
                if 0 <= gy < self.height and 0 <= gx < self.width:
                    if self.grid[gy][gx] == CellType.CONCESSION.value:
                        # Visit concession then head to exit
                        agent.goal = random.choice(self.exits)
                        start_int = Point(x=float(int(agent.pos.x)), y=float(int(agent.pos.y)))
                        goal_int = Point(x=float(int(agent.goal.x)), y=float(int(agent.goal.y)))
                        agent.path = a_star_search(self.grid, start_int, goal_int)
                        agent.rerouted = False
                    else:
                        agent.status = "arrived"
                else:
                    agent.status = "arrived"
                    
        # Remove arrived agents
        self.agents = [a for a in self.agents if a.status != "arrived"]
        
        # 3. Density analysis
        raw_density = self.density_analyzer.compute_raw_density(self.agents)
        self.density_heatmap = self.density_analyzer.get_density_grid_for_heatmap(self.agents)
        self.flow_vectors = self.density_analyzer.get_flow_vectors_for_viz(self.agents)
        self.zone_densities = self.density_analyzer.classify_zones(raw_density)
        self.density_analyzer.update_history(raw_density)
        
        # 4. Detect bottlenecks & hazards
        self.hazard_zones = self.density_analyzer.detect_hazards(raw_density)
        self.bottlenecks = [
            Point(x=float(hz.cell_x), y=float(hz.cell_y)) 
            for hz in self.hazard_zones 
            if hz.severity in (AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY)
        ]
        
        # 5. Generate alerts
        new_alerts = self.density_analyzer.generate_alerts(self.hazard_zones)
        self.alerts = new_alerts[:self.max_alerts]
        self.alert_history.extend(new_alerts)
        if len(self.alert_history) > 200:
            self.alert_history = self.alert_history[-200:]
        
        # 6. Dynamic rerouting
        self._handle_rerouting(raw_density)
        
        # 7. Auto gate actuation
        self._auto_gate_control(raw_density)
        
        # 8. Update digital signage
        self._update_signage()
        
        # 9. Run prediction forecast periodically
        if self.tick_count % self.prediction_interval == 0 and len(self.agents) > 5:
            self.predictions = self.prediction_engine.forecast(
                self.agents, self.grid, self.walls_cache,
                self.gates, self.exits,
                sim_time_sec=self.sim_time,
            )
    
    def _handle_rerouting(self, raw_density: np.ndarray):
        """Reroute agents whose paths intersect critical zones."""
        if not self.bottlenecks:
            self.suggested_routes.clear()
            self.active_reroutes.clear()
            for agent in self.agents:
                agent.color = "#3b82f6"
                agent.rerouted = False
            return
        
        bottleneck_set = set()
        for p in self.bottlenecks:
            bottleneck_set.add((int(p.x), int(p.y)))
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    bottleneck_set.add((int(p.x) + dx, int(p.y) + dy))
        
        reroute_count = 0
        self.active_reroutes.clear()
        
        for agent in self.agents:
            if agent.status == "arrived":
                continue
            
            # Check if path goes through bottleneck
            path_through_bottleneck = False
            for p in agent.path[:15]:
                if (int(p.x), int(p.y)) in bottleneck_set:
                    path_through_bottleneck = True
                    break
            
            if path_through_bottleneck:
                start_int = Point(x=float(int(agent.pos.x)), y=float(int(agent.pos.y)))
                
                # Use density-weighted pathfinding to find best alternative
                alt_routes = compute_alternative_routes(
                    self.grid, raw_density, start_int, self.exits
                )
                
                if alt_routes:
                    best_goal, best_path, best_cost = alt_routes[0]
                    if best_path:
                        agent.path = best_path[1:] if len(best_path) > 1 else best_path
                        agent.goal = best_goal
                        self.suggested_routes[agent.id] = best_path
                        agent.color = "#f59e0b"  # Amber for rerouted
                        agent.rerouted = True
                        reroute_count += 1
                        
                        # Record active reroute
                        if len(self.active_reroutes) < 10:
                            self.active_reroutes.append(RerouteRecommendation(
                                reroute_id=f"reroute_{agent.id}",
                                origin_zone=f"cell_{int(agent.pos.x)}_{int(agent.pos.y)}",
                                congested_zone=f"bottleneck_zone",
                                target_zone=f"cell_{int(best_goal.x)}_{int(best_goal.y)}",
                                path_polyline=best_path[:10],
                            ))
            elif not agent.rerouted:
                agent.color = "#3b82f6"
    
    def _auto_gate_control(self, raw_density: np.ndarray):
        """Automatically throttle gates that feed into congested downstream zones."""
        for gate_state in self.gate_objects:
            gx, gy = int(gate_state.position.x), int(gate_state.position.y)
            
            # Check density in downstream area (5-cell radius ahead of gate)
            max_downstream_density = 0.0
            for dx in range(1, 8):
                for dy in range(-2, 3):
                    nx, ny = gx + dx, gy + dy
                    if 0 <= nx < self.width and 0 <= ny < self.height:
                        max_downstream_density = max(max_downstream_density, raw_density[ny][nx])
            
            # Auto-throttle logic
            if max_downstream_density > 3.5:
                gate_state.action = GateAction.THROTTLE_FLOW
                gate_state.target_rate_per_sec = 0.5
                gate_state.status = "THROTTLED"
            elif max_downstream_density > 2.0:
                gate_state.action = GateAction.THROTTLE_FLOW
                gate_state.target_rate_per_sec = 2.0
                gate_state.status = "RESTRICTED"
            else:
                gate_state.action = GateAction.OPEN_FULL
                gate_state.target_rate_per_sec = 3.5
                gate_state.status = "OPEN"
    
    def _update_signage(self):
        """Update digital signage based on current congestion state."""
        has_bottleneck = len(self.bottlenecks) > 0
        
        for sign in self.signs:
            if has_bottleneck:
                # Check if any bottleneck is near this sign's default direction
                sign.active = True
                # Simple logic: redirect traffic away from bottlenecks
                closest_bottleneck = None
                min_dist = float('inf')
                for bp in self.bottlenecks:
                    d = (sign.position.x - bp.x)**2 + (sign.position.y - bp.y)**2
                    if d < min_dist:
                        min_dist = d
                        closest_bottleneck = bp
                
                if closest_bottleneck and min_dist < 100:
                    # Redirect: if bottleneck is to the right, suggest going up/down
                    if closest_bottleneck.x > sign.position.x:
                        if closest_bottleneck.y > sign.position.y:
                            sign.direction = "UP"
                            sign.message = "↑ Divert: Use Upper Corridor"
                        else:
                            sign.direction = "DOWN"
                            sign.message = "↓ Divert: Use Lower Corridor"
                    else:
                        sign.direction = "RIGHT"
                        sign.message = "→ Continue: Path Clear"
            else:
                sign.active = False
                sign.message = "→ Normal Flow"
                
    def set_sim_speed(self, speed: float):
        """Set simulation speed multiplier."""
        self.sim_speed = max(0.0, min(10.0, speed))
    
    def actuate_gate(self, gate_id: str, action: str, rate: float = 3.5):
        """Manually control a gate."""
        for gate in self.gate_objects:
            if gate.gate_id == gate_id:
                gate.action = GateAction(action)
                gate.target_rate_per_sec = rate
                gate.status = "MANUAL_" + action
                break
                
    def get_state(self) -> dict:
        total_rerouted = sum(1 for a in self.agents if a.rerouted)
        densities = [zd.density for zd in self.zone_densities] if self.zone_densities else [0.0]
        
        return SimulationState(
            agents=self.agents,
            real_agents=self.real_agents,
            bottlenecks=self.bottlenecks,
            suggested_routes=self.suggested_routes,
            zone_densities=self.zone_densities[:100],  # Cap payload
            hazard_zones=self.hazard_zones[:30],
            alerts=self.alerts[:20],
            gates=self.gate_objects,
            signs=self.signs,
            predictions=self.predictions[:10],
            total_agents=len(self.agents),
            total_rerouted=total_rerouted,
            avg_density=round(sum(densities) / max(len(densities), 1), 2),
            max_density=round(max(densities), 2) if densities else 0.0,
            sim_time_sec=round(self.sim_time, 1),
            sim_speed=self.sim_speed,
        ).model_dump()
    
    def get_heatmap_data(self) -> dict:
        """Return heatmap + flow vector data for frontend rendering."""
        return {
            "heatmap": self.density_heatmap,
            "flow_vectors": self.flow_vectors,
            "width": self.width,
            "height": self.height,
        }

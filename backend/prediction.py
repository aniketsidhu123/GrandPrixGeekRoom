"""
Prediction Engine — 15-30 minute crowd forecast via cloned forward simulation.

Clones the current simulation state (agent positions, velocities, goals) and
runs it forward in accelerated time to identify future bottlenecks before they form.
"""

import copy
import numpy as np
from typing import List, Optional
from .models import (
    Agent, Point, PredictionSnapshot, HazardZone, ZoneDensity,
    RerouteRecommendation, FruinLoS, AlertSeverity
)
from .density import DensityAnalyzer, classify_fruin


class PredictionEngine:
    """
    Predictive forecast engine for crowd simulation.
    
    Takes a snapshot of the current simulation state and runs an accelerated
    clone forward to project density distributions 15-30 minutes into the future.
    """
    
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.density_analyzer = DensityAnalyzer(width, height)
        
        # Forecast configuration
        self.horizon_minutes = 15
        self.snapshot_interval_sec = 60   # Output a snapshot every 60 sim seconds
        self.sim_dt = 0.2                  # Larger dt for faster forecasting
        self.steps_per_snapshot = 30       # sim steps between snapshots
        
        # Ingress rate estimation (agents/second based on recent history)
        self.recent_spawn_counts: List[int] = []
        self.estimated_ingress_rate = 0.5  # agents/sec default
        
    def update_ingress_estimate(self, agents_spawned: int):
        """Track recent spawning to estimate future ingress rate."""
        self.recent_spawn_counts.append(agents_spawned)
        if len(self.recent_spawn_counts) > 20:
            self.recent_spawn_counts.pop(0)
        
        if self.recent_spawn_counts:
            self.estimated_ingress_rate = sum(self.recent_spawn_counts) / max(len(self.recent_spawn_counts), 1)
    
    def forecast(
        self, 
        agents: List[Agent], 
        grid: List[List[str]],
        walls: List,
        gates: List[Point],
        exits: List[Point],
    ) -> List[PredictionSnapshot]:
        """
        Run a cloned simulation forward and return prediction snapshots.
        
        Each snapshot contains:
        - Forecasted zone densities
        - Predicted hazard zones
        - Maximum density across the venue
        - Count of critical zones
        """
        # Clone agents (deep copy to avoid mutating live state)
        clone_agents = []
        for a in agents:
            if a.status != "arrived":
                clone_agents.append(Agent(
                    id=a.id,
                    pos=Point(x=a.pos.x, y=a.pos.y),
                    vel=Point(x=a.vel.x, y=a.vel.y),
                    goal=Point(x=a.goal.x, y=a.goal.y),
                    path=[Point(x=p.x, y=p.y) for p in a.path],
                    status=a.status,
                    radius=a.radius,
                    mass=a.mass,
                    desired_speed=a.desired_speed,
                    profile=a.profile,
                ))
        
        snapshots = []
        total_steps = int((self.horizon_minutes * 60) / self.sim_dt)
        snapshot_every = max(1, int(self.snapshot_interval_sec / self.sim_dt))
        
        for step in range(total_steps):
            # Simplified physics update (no full SFM — just desire force for speed)
            self._fast_forward_step(clone_agents, grid)
            
            # Remove agents that reached their goals
            clone_agents = [a for a in clone_agents if a.status != "arrived"]
            
            # Take snapshot at intervals
            if (step + 1) % snapshot_every == 0:
                time_offset = (step + 1) * self.sim_dt
                snapshot = self._capture_snapshot(clone_agents, time_offset)
                snapshots.append(snapshot)
                
                # Cap snapshots to keep payload manageable
                if len(snapshots) >= 15:
                    break
        
        return snapshots
    
    def _fast_forward_step(self, agents: List[Agent], grid: List[List[str]]):
        """
        Simplified physics step for fast forecasting.
        Uses only desire force (no inter-agent repulsion) for speed.
        """
        for agent in agents:
            if agent.status == "arrived":
                continue
            
            # Determine target
            if agent.path:
                target = np.array([agent.path[0].x + 0.5, agent.path[0].y + 0.5])
            else:
                target = np.array([agent.goal.x + 0.5, agent.goal.y + 0.5])
            
            pos = np.array([agent.pos.x, agent.pos.y])
            direction = target - pos
            dist = np.linalg.norm(direction)
            
            if dist > 0.1:
                e_i = direction / dist
                desired_vel = agent.desired_speed * e_i
                # Simple velocity update with relaxation
                vel = np.array([agent.vel.x, agent.vel.y])
                new_vel = vel + (desired_vel - vel) * 0.3  # Relaxation factor
                
                # Cap speed
                speed = np.linalg.norm(new_vel)
                if speed > agent.desired_speed * 1.3:
                    new_vel = (new_vel / speed) * agent.desired_speed
                
                agent.vel = Point(x=float(new_vel[0]), y=float(new_vel[1]))
                new_pos = pos + new_vel * self.sim_dt
                
                # Boundary clamping
                new_pos[0] = max(0, min(self.width - 1, new_pos[0]))
                new_pos[1] = max(0, min(self.height - 1, new_pos[1]))
                
                # Wall collision check
                nx, ny = int(new_pos[0]), int(new_pos[1])
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if grid[ny][nx] == "wall":
                        continue  # Don't move into walls
                
                agent.pos = Point(x=float(new_pos[0]), y=float(new_pos[1]))
            
            # Check path progress
            if agent.path:
                next_node = agent.path[0]
                d2 = (agent.pos.x - (next_node.x + 0.5))**2 + (agent.pos.y - (next_node.y + 0.5))**2
                if d2 < 0.5:
                    agent.path.pop(0)
            
            # Check goal arrival
            d_goal = (agent.pos.x - (agent.goal.x + 0.5))**2 + (agent.pos.y - (agent.goal.y + 0.5))**2
            if d_goal < 1.5:
                agent.status = "arrived"
    
    def _capture_snapshot(self, agents: List[Agent], time_offset: float) -> PredictionSnapshot:
        """Capture density state at a forecast timestep."""
        raw_density = self.density_analyzer.compute_raw_density(agents)
        hazards = self.density_analyzer.detect_hazards(raw_density)
        zones = self.density_analyzer.classify_zones(raw_density)
        
        max_d = float(raw_density.max()) if raw_density.size > 0 else 0.0
        critical_count = sum(1 for h in hazards if h.severity in (
            AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY
        ))
        
        return PredictionSnapshot(
            timestamp_offset_sec=round(time_offset, 1),
            hazard_zones=hazards[:20],  # Cap to prevent massive payloads
            zone_densities=zones[:50],
            max_density=round(max_d, 2),
            critical_zone_count=critical_count,
        )

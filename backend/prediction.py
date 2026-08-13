"""
Prediction Engine — 15-30 minute crowd forecast via cloned forward simulation.

Clones the current simulation state (agent positions, velocities, goals) and
runs it forward in accelerated time to identify future bottlenecks before they
form.

Two things make the forecast trustworthy rather than decorative:

  1. The fast-forward keeps crowd repulsion. Without it, cloned agents stack on
     the same waypoint, per-cell density saturates, and every horizon step
     reports the same number — a constant, not a prediction.

     Repulsion here is a density-gradient (pressure) term rather than the
     pairwise exponential the live physics uses. Agents are pushed down the
     local density gradient, which is the continuum-crowds formulation. It is
     O(agents + cells) instead of O(agents²), so a 15-minute horizon over a few
     hundred people stays in the tens of milliseconds. Fine-grained contact
     mechanics are left to the live Social Force Model; the forecast only needs
     the aggregate density field to be right.

  2. New arrivals are injected during the forward run using a measured arrival
     curve (see `ingress.ArrivalModel`). Real bottlenecks form because people
     keep arriving; a forecast that only drains the current crowd can only ever
     predict the congestion getting better.

The step loop is vectorised over agents so the whole horizon stays well under a
second and can run inside a thread executor without stalling the event loop.
"""

import numpy as np
from typing import List, Optional

from .models import (
    Agent, Point, PredictionSnapshot, HazardZone, ZoneDensity,
    RerouteRecommendation, FruinLoS, AlertSeverity
)
from .density import DensityAnalyzer, classify_fruin
from .pathfinding import a_star_search


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
        self.sim_dt = 0.5                 # Coarser than live physics; still sub-metre steps
        self.max_snapshots = 15

        # Crowd repulsion, as a density-gradient pressure term.
        self.pressure_strength = 1.5      # velocity added per unit density gradient
        self.max_pressure = 1.2           # m/s cap, so pressure cannot outrun walking

        # Hard occupancy limit, people per 1 m² cell. Fruin's LoS F begins at
        # 3.5 and jam density for standing crowds is around 5; above that bodies
        # simply do not fit. Without this bound the forecast happily stacks
        # dozens of people on one waypoint and reports impossible densities.
        self.cell_capacity = 5.0

        # Ingress. Set by the simulation engine; None disables arrival injection.
        self.arrival_model = None
        self.gate_positions: List[Point] = []

        # Legacy ingress estimate, kept for the vision pipeline path.
        self.recent_spawn_counts: List[int] = []
        self.estimated_ingress_rate = 0.5  # agents/sec default

    # ─────────────────────────────────────────────
    # INGRESS
    # ─────────────────────────────────────────────

    def set_arrival_model(self, model, gates: List[Point]):
        """Attach a measured arrival curve used to inject future arrivals."""
        self.arrival_model = model
        self.gate_positions = list(gates)

    def update_ingress_estimate(self, agents_spawned: int):
        """Track recent spawning to estimate future ingress rate."""
        self.recent_spawn_counts.append(agents_spawned)
        if len(self.recent_spawn_counts) > 20:
            self.recent_spawn_counts.pop(0)
        if self.recent_spawn_counts:
            self.estimated_ingress_rate = (
                sum(self.recent_spawn_counts) / max(len(self.recent_spawn_counts), 1)
            )

    # ─────────────────────────────────────────────
    # FORECAST
    # ─────────────────────────────────────────────

    def forecast(
        self,
        agents: List[Agent],
        grid: List[List[str]],
        walls: List,
        gates: List[Point],
        exits: List[Point],
        sim_time_sec: float = 0.0,
    ) -> List[PredictionSnapshot]:
        """
        Run a cloned simulation forward and return prediction snapshots.

        Each snapshot contains forecast zone densities, predicted hazard zones,
        the venue-wide maximum density and a count of critical zones.
        """
        live = [a for a in agents if a.status != "arrived"]
        if not live:
            return []

        wall_mask = self._build_wall_mask(grid)
        state = _ForecastState.from_agents(live, self.width, self.height)

        # One reusable route per gate, so injected arrivals follow corridors
        # instead of walking through walls. Computed once, not per agent.
        spawn_routes = self._build_spawn_routes(grid, gates, exits)

        total_steps = int((self.horizon_minutes * 60) / self.sim_dt)
        snapshot_every = max(1, int(self.snapshot_interval_sec / self.sim_dt))

        snapshots: List[PredictionSnapshot] = []
        # People who have arrived but could not get through a gate yet. They
        # queue outside rather than materialising on top of the crowd already
        # standing in the gate cell.
        backlog = 0.0

        for step in range(total_steps):
            self._step(state, wall_mask)

            if self.arrival_model is not None and spawn_routes:
                elapsed = (step + 1) * self.sim_dt
                backlog += self.arrival_model.arrivals_in(
                    sim_time_sec + elapsed, self.sim_dt
                )
                wanting_in = int(backlog)
                if wanting_in:
                    admitted = state.inject(
                        wanting_in, spawn_routes, self.cell_capacity
                    )
                    backlog -= admitted

            state.drop_arrived()

            if (step + 1) % snapshot_every == 0:
                time_offset = (step + 1) * self.sim_dt
                snapshots.append(self._capture_snapshot(state, time_offset))
                if len(snapshots) >= self.max_snapshots:
                    break

        return snapshots

    def _build_wall_mask(self, grid: List[List[str]]) -> np.ndarray:
        """Boolean (height, width) mask, True where a cell is a wall."""
        mask = np.zeros((self.height, self.width), dtype=bool)
        for y in range(min(self.height, len(grid))):
            row = grid[y]
            for x in range(min(self.width, len(row))):
                if row[x] == "wall":
                    mask[y][x] = True
        return mask

    def _build_spawn_routes(self, grid, gates: List[Point], exits: List[Point]):
        """Precompute one gate -> exit route per gate for injected arrivals."""
        routes = []
        if not gates or not exits:
            return routes
        for i, gate in enumerate(gates):
            target = exits[i % len(exits)]
            start = Point(x=float(int(gate.x)), y=float(int(gate.y)))
            goal = Point(x=float(int(target.x)), y=float(int(target.y)))
            path = a_star_search(grid, start, goal)
            if path:
                routes.append((gate, target, path))
        return routes

    # ─────────────────────────────────────────────
    # PHYSICS STEP (vectorised)
    # ─────────────────────────────────────────────

    def _step(self, s: "_ForecastState", wall_mask: np.ndarray):
        """
        Advance every agent one timestep.

        Desire force pulls each agent toward its next waypoint; pairwise
        repulsion pushes overlapping agents apart so density stays physical.
        """
        n = s.count
        if n == 0:
            return

        pos = s.pos[:n]
        vel = s.vel[:n]

        target = s.current_targets()
        occupancy = s.density_grid()

        # ── Desire force: steer toward the next waypoint ──
        delta = target - pos
        dist = np.linalg.norm(delta, axis=1, keepdims=True)
        safe = np.maximum(dist, 1e-6)
        desired = (delta / safe) * s.desired_speed[:n, None]
        new_vel = vel + (desired - vel) * 0.3   # relaxation toward desired velocity

        # ── Repulsion: push down the density gradient, away from crowding ──
        if n > 1:
            new_vel = new_vel + self._density_pressure(occupancy, pos)

        # ── Speed cap ──
        speed = np.linalg.norm(new_vel, axis=1, keepdims=True)
        cap = s.desired_speed[:n, None] * 1.3
        too_fast = (speed > cap).ravel()
        if np.any(too_fast):
            new_vel[too_fast] = (
                new_vel[too_fast] / np.maximum(speed[too_fast], 1e-6) * cap[too_fast]
            )

        # ── Integrate, clamped to the venue bounds ──
        new_pos = pos + new_vel * self.sim_dt
        np.clip(new_pos[:, 0], 0, self.width - 1, out=new_pos[:, 0])
        np.clip(new_pos[:, 1], 0, self.height - 1, out=new_pos[:, 1])

        # ── Wall collision: reject moves that land inside a wall cell ──
        cx = new_pos[:, 0].astype(np.int32)
        cy = new_pos[:, 1].astype(np.int32)
        blocked = wall_mask[cy, cx]

        # ── Capacity: a full cell cannot accept anyone new. Agents already
        #    inside one may still leave, so only cell-crossing moves are tested.
        #
        #    Every agent moves in the same step, so testing each one against the
        #    pre-move occupancy would let a crowd pour into one cell together,
        #    all of them seeing it as empty. Instead the intended arrivals are
        #    counted per cell, and where they exceed the remaining headroom the
        #    surplus is turned away — chosen at random, since within a step
        #    there is no meaningful ordering between them.
        src_x = pos[:, 0].astype(np.int32)
        src_y = pos[:, 1].astype(np.int32)
        crossing = (cx != src_x) | (cy != src_y)

        dest = (cy * self.width + cx).astype(np.int64)
        incoming = np.bincount(dest[crossing], minlength=self.width * self.height)
        headroom = np.maximum(self.cell_capacity - occupancy.ravel(), 0.0)

        with np.errstate(divide="ignore", invalid="ignore"):
            admit_fraction = np.where(incoming > 0, headroom / incoming, 1.0)
        turn_away_p = np.clip(1.0 - admit_fraction, 0.0, 1.0)

        over_capacity = crossing & (np.random.random(n) < turn_away_p[dest])

        rejected = blocked | over_capacity
        if np.any(rejected):
            new_pos[rejected] = pos[rejected]
            new_vel[rejected] = 0.0

        s.pos[:n] = new_pos
        s.vel[:n] = new_vel

        s.advance_paths()

    def _density_pressure(self, occupancy: np.ndarray, pos: np.ndarray) -> np.ndarray:
        """
        Velocity contribution pushing each agent away from local crowding.

        The occupancy grid is blurred so the gradient is smooth rather than
        stepping between whole-person counts, then each agent samples the
        negative gradient at its own cell. Cost is independent of how many
        agent pairs exist, which is what keeps the horizon affordable.
        """
        field = _blur3(occupancy)
        grad_y, grad_x = np.gradient(field)

        cx = np.clip(pos[:, 0].astype(np.int32), 0, self.width - 1)
        cy = np.clip(pos[:, 1].astype(np.int32), 0, self.height - 1)

        push = np.empty_like(pos)
        push[:, 0] = -grad_x[cy, cx]
        push[:, 1] = -grad_y[cy, cx]
        push *= self.pressure_strength

        # Cap so a dense cell cannot fling agents across the venue
        mag = np.linalg.norm(push, axis=1, keepdims=True)
        over = (mag > self.max_pressure).ravel()
        if np.any(over):
            push[over] = push[over] / np.maximum(mag[over], 1e-6) * self.max_pressure
        return push

    # ─────────────────────────────────────────────
    # SNAPSHOT
    # ─────────────────────────────────────────────

    def _capture_snapshot(self, s: "_ForecastState", time_offset: float) -> PredictionSnapshot:
        """Capture density state at a forecast timestep."""
        raw_density = s.density_grid()
        hazards = self.density_analyzer.detect_hazards(raw_density)
        zones = self.density_analyzer.classify_zones(raw_density)

        max_d = float(raw_density.max()) if raw_density.size > 0 else 0.0
        critical_count = sum(
            1 for h in hazards
            if h.severity in (AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY)
        )

        return PredictionSnapshot(
            timestamp_offset_sec=round(time_offset, 1),
            hazard_zones=hazards[:20],   # Cap to prevent massive payloads
            zone_densities=zones[:50],
            max_density=round(max_d, 2),
            critical_zone_count=critical_count,
        )


def _blur3(field: np.ndarray) -> np.ndarray:
    """
    Cheap separable 3-tap blur, used to smooth the occupancy grid before
    differentiating it. Without this the gradient steps between whole-person
    counts and agents jitter instead of flowing.
    """
    padded = np.pad(field, 1, mode="edge")
    horizontal = padded[:, :-2] + padded[:, 1:-1] + padded[:, 2:]
    return (horizontal[:-2] + horizontal[1:-1] + horizontal[2:]) / 9.0


class _ForecastState:
    """
    Struct-of-arrays agent state for the forecast run.

    Everything the step loop touches — positions, velocities, waypoint queues —
    lives in numpy arrays so no per-agent Python loop runs inside the horizon.
    Waypoint queues are padded to a fixed width rather than kept ragged, which
    is what allows the target lookup and arrival test to be vectorised.

    Arrays are over-allocated and tracked by `count`, so injected arrivals do
    not reallocate on every step.
    """

    GROWTH = 512
    MAX_PATH = 96   # A* paths on a 40x30 grid stay well under this

    def __init__(self, capacity: int, width: int, height: int):
        self.width = width
        self.height = height
        self.count = 0

        self.pos = np.zeros((capacity, 2), dtype=np.float64)
        self.vel = np.zeros((capacity, 2), dtype=np.float64)
        self.goal = np.zeros((capacity, 2), dtype=np.float64)
        self.desired_speed = np.full(capacity, 1.3, dtype=np.float64)
        self.radius = np.full(capacity, 0.3, dtype=np.float64)

        self.path_xy = np.zeros((capacity, self.MAX_PATH, 2), dtype=np.float32)
        self.path_len = np.zeros(capacity, dtype=np.int32)
        self.path_idx = np.zeros(capacity, dtype=np.int32)

    @classmethod
    def from_agents(cls, agents: List[Agent], width: int, height: int) -> "_ForecastState":
        s = cls(len(agents) + cls.GROWTH, width, height)
        for a in agents:
            s._append(
                pos=(a.pos.x, a.pos.y),
                vel=(a.vel.x, a.vel.y),
                goal=(a.goal.x, a.goal.y),
                desired_speed=a.desired_speed,
                radius=a.radius,
                path=[(p.x, p.y) for p in a.path],
            )
        return s

    def _ensure_capacity(self, extra: int):
        if self.count + extra <= self.pos.shape[0]:
            return
        new_cap = self.count + extra + self.GROWTH
        n = self.count
        for name in ("pos", "vel", "goal"):
            arr = getattr(self, name)
            grown = np.zeros((new_cap, 2), dtype=arr.dtype)
            grown[:n] = arr[:n]
            setattr(self, name, grown)
        for name, fill in (("desired_speed", 1.3), ("radius", 0.3)):
            arr = getattr(self, name)
            grown = np.full(new_cap, fill, dtype=arr.dtype)
            grown[:n] = arr[:n]
            setattr(self, name, grown)
        grown_xy = np.zeros((new_cap, self.MAX_PATH, 2), dtype=np.float32)
        grown_xy[:n] = self.path_xy[:n]
        self.path_xy = grown_xy
        for name in ("path_len", "path_idx"):
            arr = getattr(self, name)
            grown = np.zeros(new_cap, dtype=np.int32)
            grown[:n] = arr[:n]
            setattr(self, name, grown)

    def _append(self, pos, vel, goal, desired_speed, radius, path):
        self._ensure_capacity(1)
        i = self.count
        self.pos[i] = pos
        self.vel[i] = vel
        self.goal[i] = goal
        self.desired_speed[i] = desired_speed
        self.radius[i] = radius

        length = min(len(path), self.MAX_PATH)
        if length:
            self.path_xy[i, :length] = path[:length]
        self.path_len[i] = length
        self.path_idx[i] = 0
        self.count += 1

    def inject(self, n: int, spawn_routes, cell_capacity: float) -> int:
        """
        Admit up to `n` new arrivals across the gate routes.

        A gate cell that is already at capacity cannot take anyone: those people
        are still queueing outside. Returns how many actually got in, so the
        caller can carry the rest forward instead of losing them.
        """
        self._ensure_capacity(n)
        occupancy = self.density_grid()
        admitted = 0

        for k in range(n):
            gate, target, path = spawn_routes[k % len(spawn_routes)]
            gx = int(np.clip(gate.x, 0, self.width - 1))
            gy = int(np.clip(gate.y, 0, self.height - 1))
            if occupancy[gy, gx] >= cell_capacity:
                continue   # gate is jammed; this arrival waits outside

            occupancy[gy, gx] += 1.0
            self._append(
                pos=(gate.x + np.random.uniform(0.1, 0.9),
                     gate.y + np.random.uniform(-0.4, 0.4)),
                vel=(0.0, 0.0),
                goal=(target.x, target.y),
                desired_speed=float(np.random.uniform(1.0, 1.4)),
                radius=0.3,
                path=[(p.x, p.y) for p in path],
            )
            admitted += 1

        return admitted

    def _waypoints(self):
        """(has_waypoint, waypoint_centres) for every live agent."""
        n = self.count
        idx = self.path_idx[:n]
        has = idx < self.path_len[:n]
        safe = np.where(has, idx, 0)
        wp = self.path_xy[np.arange(n), safe].astype(np.float64) + 0.5
        return has, wp

    def current_targets(self) -> np.ndarray:
        """Next waypoint per agent, falling back to the final goal."""
        n = self.count
        has, wp = self._waypoints()
        return np.where(has[:, None], wp, self.goal[:n] + 0.5)

    def advance_paths(self):
        """Advance past waypoints that have been reached."""
        n = self.count
        if n == 0:
            return
        has, wp = self._waypoints()
        delta = self.pos[:n] - wp
        reached = has & (np.einsum("ij,ij->i", delta, delta) < 0.5)
        self.path_idx[:n] += reached

    def drop_arrived(self):
        """Remove agents that reached their goal, compacting the arrays."""
        n = self.count
        if n == 0:
            return
        delta = self.pos[:n] - (self.goal[:n] + 0.5)
        keep = np.einsum("ij,ij->i", delta, delta) >= 1.5
        if np.all(keep):
            return

        kept = np.flatnonzero(keep)
        m = kept.size
        for name in ("pos", "vel", "goal", "desired_speed", "radius",
                     "path_xy", "path_len", "path_idx"):
            arr = getattr(self, name)
            arr[:m] = arr[kept]
        self.count = m

    def density_grid(self) -> np.ndarray:
        """People-per-cell grid. Cells are 1 m², so this is people/m²."""
        grid = np.zeros((self.height, self.width), dtype=np.float64)
        n = self.count
        if n == 0:
            return grid
        cx = np.clip(self.pos[:n, 0].astype(np.int32), 0, self.width - 1)
        cy = np.clip(self.pos[:n, 1].astype(np.int32), 0, self.height - 1)
        np.add.at(grid, (cy, cx), 1.0)
        return grid

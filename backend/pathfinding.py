"""
Pathfinding — Density-weighted A* with exponential congestion cost formula.

Implements the spec's dynamic edge weight formula:
    W_e = L_e * [1 + α * (ρ_e / ρ_critical)^β]

where α=2.0 and β=4.0. As zone density approaches ρ_critical,
path cost rises steeply, causing the router to instantly divert
agents to less congested alternate routes.
"""

import heapq
import numpy as np
from typing import List, Tuple, Dict, Optional
from .models import Point, CellType


# Density-weighted pathfinding constants (per spec §4.3)
ALPHA = 2.0         # Congestion cost multiplier
BETA = 4.0          # Congestion cost exponent
RHO_CRITICAL = 3.5  # Critical density threshold (people/m²)


def heuristic(a: Point, b: Point) -> float:
    """Manhattan distance heuristic for A*."""
    return abs(a.x - b.x) + abs(a.y - b.y)


def density_weighted_cost(
    base_cost: float, 
    density: float, 
    alpha: float = ALPHA, 
    beta: float = BETA,
    rho_critical: float = RHO_CRITICAL
) -> float:
    """
    Compute density-weighted edge cost per spec formula:
    W_e = L_e * [1 + α * (ρ_e / ρ_critical)^β]
    
    As density approaches ρ_critical, cost rises exponentially.
    At density = 0, cost = base_cost (no penalty).
    At density = ρ_critical, cost = base_cost * (1 + α) = base_cost * 3.0
    """
    if density <= 0:
        return base_cost
    
    ratio = min(density / rho_critical, 2.0)  # Cap ratio to prevent infinity
    penalty = alpha * (ratio ** beta)
    return base_cost * (1.0 + penalty)


def a_star_search(
    grid: List[List[str]],
    start: Point,
    goal: Point,
    congestion_map: Dict[Point, float] = None,
    density_grid: Optional[np.ndarray] = None,
) -> List[Point]:
    """
    A* pathfinding with density-weighted edge costs.
    
    Args:
        grid: 2D grid of CellType values
        start: Starting cell
        goal: Target cell
        congestion_map: Legacy flat penalty map (still supported)
        density_grid: Full density grid for exponential cost formula.
                      If provided, uses W_e = L_e * [1 + α(ρ/ρ_c)^β].
                      Takes priority over congestion_map.
    """
    if congestion_map is None:
        congestion_map = {}
        
    width = len(grid[0])
    height = len(grid)
    
    # Validate bounds
    if not (0 <= int(start.x) < width and 0 <= int(start.y) < height):
        return []
    if not (0 <= int(goal.x) < width and 0 <= int(goal.y) < height):
        return []
    
    frontier = []
    counter = 0  # Tie-breaker for heap ordering
    heapq.heappush(frontier, (0, counter, start))
    came_from: Dict[Point, Optional[Point]] = {}
    cost_so_far: Dict[Point, float] = {}
    
    came_from[start] = None
    cost_so_far[start] = 0
    
    # 8-way movement
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]
    
    while frontier:
        _, _, current = heapq.heappop(frontier)
        
        if current == goal:
            break
            
        for dx, dy in directions:
            next_x = int(current.x + dx)
            next_y = int(current.y + dy)
            
            # Check bounds
            if 0 <= next_x < width and 0 <= next_y < height:
                next_p = Point(x=float(next_x), y=float(next_y))
                
                # Check walls
                if grid[next_y][next_x] == CellType.WALL.value or grid[next_y][next_x] == CellType.WALL:
                    continue
                    
                # Base movement cost (diagonal = √2, straight = 1)
                base_cost = 1.414 if dx != 0 and dy != 0 else 1.0
                
                # Apply density-weighted cost
                if density_grid is not None:
                    cell_density = float(density_grid[next_y][next_x])
                    move_cost = density_weighted_cost(base_cost, cell_density)
                else:
                    # Legacy: flat penalty from congestion_map
                    penalty = congestion_map.get(next_p, 0.0)
                    move_cost = base_cost + penalty
                
                new_cost = cost_so_far[current] + move_cost
                
                if next_p not in cost_so_far or new_cost < cost_so_far[next_p]:
                    cost_so_far[next_p] = new_cost
                    priority = new_cost + heuristic(next_p, goal)
                    counter += 1
                    heapq.heappush(frontier, (priority, counter, next_p))
                    came_from[next_p] = current
                    
    # Reconstruct path
    path = []
    if goal in came_from:
        curr = goal
        while curr is not None and curr != start:
            path.append(curr)
            curr = came_from.get(curr)
        path.reverse()
    
    return path


def compute_alternative_routes(
    grid: List[List[str]],
    density_grid: np.ndarray,
    start: Point,
    goals: List[Point],
) -> List[Tuple[Point, List[Point], float]]:
    """
    Compute paths to all possible goals and rank by total density-weighted cost.
    Returns list of (goal, path, cost) sorted by cost ascending.
    
    Used for smart rerouting: when the primary exit is congested,
    find the least-congested alternate exit.
    """
    routes = []
    
    for goal in goals:
        path = a_star_search(grid, start, goal, density_grid=density_grid)
        if path:
            # Compute total path cost
            total_cost = 0.0
            for p in path:
                px, py = int(p.x), int(p.y)
                if 0 <= py < density_grid.shape[0] and 0 <= px < density_grid.shape[1]:
                    cell_density = float(density_grid[py][px])
                    total_cost += density_weighted_cost(1.0, cell_density)
                else:
                    total_cost += 1.0
            routes.append((goal, path, total_cost))
    
    routes.sort(key=lambda r: r[2])
    return routes


# Ensure Point is comparable for heapq
def point_lt(self, other):
    return (self.x, self.y) < (other.x, other.y)
Point.__lt__ = point_lt

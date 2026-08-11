/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   CROWD FLOW OPTIMISER — Multi-Layer Canvas Rendering Engine
   Features: Heatmap, Flow Vectors, Path Animations, Interpolation,
             Minimap, Tooltips, Zoom/Pan
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */

// ═══════════════════════════════════════════════════════════════════
// STATE
// ═══════════════════════════════════════════════════════════════════

let ws;
let gridConfig = null;
let simulationState = null;
let heatmapData = null;
let prevAgentPositions = {};
let animationFrame = 0;

const CELL_SIZE = 20;
const INTERP_FACTOR = 0.18;

// Layer visibility
const layers = {
    heatmap: true,
    flow: true,
    paths: true,
    agents: true,
    labels: false,
};

// Zoom / Pan state
let zoomLevel = 1.0;
let panX = 0;
let panY = 0;
let isDragging = false;
let dragStartX = 0;
let dragStartY = 0;

// ═══════════════════════════════════════════════════════════════════
// DOM REFS
// ═══════════════════════════════════════════════════════════════════

const canvas = document.getElementById('venue-canvas');
const ctx = canvas.getContext('2d');
const minimapCanvas = document.getElementById('minimap-canvas');
const minimapCtx = minimapCanvas.getContext('2d');
const tooltip = document.getElementById('tooltip');

const btnSpawn = document.getElementById('btn-spawn');
const btnClear = document.getElementById('btn-clear');
const crowdSizeInput = document.getElementById('crowd-size');
const crowdSizeVal = document.getElementById('crowd-size-val');
const statAgents = document.getElementById('stat-agents');
const statRerouted = document.getElementById('stat-rerouted');
const statMaxDensity = document.getElementById('stat-max-density');
const statHazards = document.getElementById('stat-hazards');
const alertList = document.getElementById('alert-list');
const simClock = document.getElementById('sim-clock');
const tickCounter = document.getElementById('tick-counter');
const avgDensityBadge = document.getElementById('avg-density');
const connectionDot = document.getElementById('connection-dot');
const connectionText = document.getElementById('connection-text');

// ═══════════════════════════════════════════════════════════════════
// COLOR UTILITIES
// ═══════════════════════════════════════════════════════════════════

function densityToColor(d, alpha = 1.0) {
    // Multi-stop gradient: green → yellow → orange → red → dark red
    if (d < 0.3) return `rgba(34, 197, 94, ${alpha * 0.1})`;
    if (d < 1.0) {
        const t = (d - 0.3) / 0.7;
        const r = Math.floor(34 + (250 - 34) * t);
        const g = Math.floor(197 + (204 - 197) * t);
        const b = Math.floor(94 + (21 - 94) * t);
        return `rgba(${r}, ${g}, ${b}, ${alpha * (0.15 + t * 0.25)})`;
    }
    if (d < 2.0) {
        const t = (d - 1.0) / 1.0;
        const r = Math.floor(250 + (249 - 250) * t);
        const g = Math.floor(204 + (115 - 204) * t);
        const b = Math.floor(21 + (22 - 21) * t);
        return `rgba(${r}, ${g}, ${b}, ${alpha * (0.4 + t * 0.2)})`;
    }
    if (d < 3.5) {
        const t = (d - 2.0) / 1.5;
        const r = Math.floor(249 + (220 - 249) * t);
        const g = Math.floor(115 + (38 - 115) * t);
        const b = Math.floor(22 + (38 - 22) * t);
        return `rgba(${r}, ${g}, ${b}, ${alpha * (0.6 + t * 0.2)})`;
    }
    return `rgba(153, 27, 27, ${alpha * 0.85})`;
}

function losToColor(los) {
    const map = {
        'A': '#22c55e', 'B': '#4ade80', 'C': '#facc15',
        'D': '#f97316', 'E': '#ef4444', 'F': '#dc2626', 'CRITICAL': '#991b1b'
    };
    return map[los] || '#64748b';
}

function getCellColor(type) {
    switch (type) {
        case 'wall': return '#334155';
        case 'gate': return '#10b981';
        case 'exit': return '#f59e0b';
        case 'concession': return '#8b5cf6';
        default: return '#0b1120';
    }
}

// ═══════════════════════════════════════════════════════════════════
// WEBSOCKET
// ═══════════════════════════════════════════════════════════════════

function initWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.hostname || 'localhost';
    const port = window.location.port || '8000';
    const wsUrl = `${protocol}//${host}:${port}/ws`;

    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        connectionDot.classList.remove('disconnected');
        connectionText.textContent = 'Live';
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);

        if (data.type === 'grid_config') {
            gridConfig = data;
            canvas.width = gridConfig.width * CELL_SIZE;
            canvas.height = gridConfig.height * CELL_SIZE;
            minimapCanvas.width = gridConfig.width * 4;
            minimapCanvas.height = gridConfig.height * 4;
            console.log('[CFO] Grid config received:', gridConfig.width, 'x', gridConfig.height, '=> canvas:', canvas.width, 'x', canvas.height);
            drawBaseLayer();
            console.log('[CFO] Base layer drawn, baseCanvas:', baseCanvas ? baseCanvas.width + 'x' + baseCanvas.height : 'null');
        } else if (data.type === 'state_update') {
            // Store previous positions for interpolation
            if (simulationState && simulationState.agents) {
                for (const a of simulationState.agents) {
                    prevAgentPositions[a.id] = { x: a.pos.x, y: a.pos.y };
                }
            }
            simulationState = data.state;
            heatmapData = data.heatmap;
            updateAllUI();
        }
    };

    ws.onclose = () => {
        connectionDot.classList.add('disconnected');
        connectionText.textContent = 'Reconnecting…';
        setTimeout(initWebSocket, 2000);
    };

    ws.onerror = () => {
        connectionDot.classList.add('disconnected');
        connectionText.textContent = 'Error';
    };
}

function sendCommand(cmd) {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify(cmd));
    }
}

// ═══════════════════════════════════════════════════════════════════
// RENDERING — Base Layer (Static: walls, gates, exits)
// ═══════════════════════════════════════════════════════════════════

// Offscreen canvas for base layer (only redraws when grid changes)
let baseCanvas = null;

function drawBaseLayer() {
    if (!gridConfig) return;

    baseCanvas = document.createElement('canvas');
    baseCanvas.width = gridConfig.width * CELL_SIZE;
    baseCanvas.height = gridConfig.height * CELL_SIZE;
    const bCtx = baseCanvas.getContext('2d');

    for (let y = 0; y < gridConfig.height; y++) {
        for (let x = 0; x < gridConfig.width; x++) {
            const cell = gridConfig.grid[y][x];
            bCtx.fillStyle = getCellColor(cell);
            bCtx.fillRect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE);

            // Subtle grid lines
            bCtx.strokeStyle = 'rgba(148, 163, 184, 0.04)';
            bCtx.lineWidth = 0.5;
            bCtx.strokeRect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE);
        }
    }

    // Draw gate/exit/concession labels
    for (let y = 0; y < gridConfig.height; y++) {
        for (let x = 0; x < gridConfig.width; x++) {
            const cell = gridConfig.grid[y][x];
            if (cell === 'gate' || cell === 'exit' || cell === 'concession') {
                bCtx.fillStyle = 'rgba(0,0,0,0.5)';
                bCtx.font = '600 7px Inter, sans-serif';
                bCtx.textAlign = 'center';
                bCtx.textBaseline = 'middle';
                const labels = { gate: 'G', exit: 'E', concession: 'C' };
                bCtx.fillStyle = 'rgba(255,255,255,0.8)';
                bCtx.fillText(labels[cell], x * CELL_SIZE + CELL_SIZE / 2, y * CELL_SIZE + CELL_SIZE / 2);
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// RENDERING — Heatmap Layer
// ═══════════════════════════════════════════════════════════════════

function drawHeatmapLayer() {
    if (!heatmapData || !heatmapData.heatmap) return;

    const hm = heatmapData.heatmap;
    for (let y = 0; y < hm.length; y++) {
        for (let x = 0; x < hm[y].length; x++) {
            const d = hm[y][x];
            if (d > 0.1) {
                ctx.fillStyle = densityToColor(d, 0.9);
                ctx.fillRect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE);
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// RENDERING — Flow Vector Layer
// ═══════════════════════════════════════════════════════════════════

function drawFlowVectors() {
    if (!heatmapData || !heatmapData.flow_vectors) return;

    const vectors = heatmapData.flow_vectors;
    ctx.lineWidth = 1.5;

    for (const v of vectors) {
        const cx = v.x * CELL_SIZE + CELL_SIZE / 2;
        const cy = v.y * CELL_SIZE + CELL_SIZE / 2;
        const angle = Math.atan2(v.vy, v.vx);
        const len = Math.min(v.mag * 8, CELL_SIZE * 0.7);

        const ex = cx + Math.cos(angle) * len;
        const ey = cy + Math.sin(angle) * len;

        // Arrow line
        const alpha = Math.min(0.7, 0.2 + v.mag * 0.4);
        ctx.strokeStyle = `rgba(96, 165, 250, ${alpha})`;
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(ex, ey);
        ctx.stroke();

        // Arrowhead
        const headLen = 4;
        ctx.fillStyle = `rgba(96, 165, 250, ${alpha})`;
        ctx.beginPath();
        ctx.moveTo(ex, ey);
        ctx.lineTo(
            ex - headLen * Math.cos(angle - 0.5),
            ey - headLen * Math.sin(angle - 0.5)
        );
        ctx.lineTo(
            ex - headLen * Math.cos(angle + 0.5),
            ey - headLen * Math.sin(angle + 0.5)
        );
        ctx.closePath();
        ctx.fill();
    }
}

// ═══════════════════════════════════════════════════════════════════
// RENDERING — Path Layer (animated reroute polylines)
// ═══════════════════════════════════════════════════════════════════

function drawPathLayer() {
    if (!simulationState || !simulationState.suggested_routes) return;

    const routes = simulationState.suggested_routes;
    const dashOffset = -(animationFrame * 0.5) % 20;

    ctx.setLineDash([6, 4]);
    ctx.lineDashOffset = dashOffset;
    ctx.lineWidth = 2;
    ctx.strokeStyle = 'rgba(245, 158, 11, 0.45)';

    const drawnPaths = new Set();

    for (const agentId in routes) {
        const path = routes[agentId];
        if (!path || path.length < 2) continue;

        // Deduplicate similar paths
        const key = path.slice(0, 3).map(p => `${Math.round(p.x)},${Math.round(p.y)}`).join('|');
        if (drawnPaths.has(key)) continue;
        drawnPaths.add(key);

        ctx.beginPath();
        ctx.moveTo(path[0].x * CELL_SIZE + CELL_SIZE / 2, path[0].y * CELL_SIZE + CELL_SIZE / 2);
        for (let i = 1; i < path.length; i++) {
            ctx.lineTo(path[i].x * CELL_SIZE + CELL_SIZE / 2, path[i].y * CELL_SIZE + CELL_SIZE / 2);
        }
        ctx.stroke();
    }

    ctx.setLineDash([]);
}

// ═══════════════════════════════════════════════════════════════════
// RENDERING — Agent Layer (circles with velocity trails)
// ═══════════════════════════════════════════════════════════════════

function drawAgents() {
    if (!simulationState || !simulationState.agents) return;

    for (const agent of simulationState.agents) {
        // Interpolate position for smooth movement
        let renderX = agent.pos.x;
        let renderY = agent.pos.y;

        const prev = prevAgentPositions[agent.id];
        if (prev) {
            renderX = prev.x + (agent.pos.x - prev.x) * INTERP_FACTOR;
            renderY = prev.y + (agent.pos.y - prev.y) * INTERP_FACTOR;
            // Update for next frame
            prevAgentPositions[agent.id] = { x: renderX, y: renderY };
        }

        const px = renderX * CELL_SIZE + CELL_SIZE / 2;
        const py = renderY * CELL_SIZE + CELL_SIZE / 2;
        const radius = CELL_SIZE / 2.8;

        // Velocity trail (subtle particle effect)
        const speed = Math.sqrt(agent.vel.x ** 2 + agent.vel.y ** 2);
        if (speed > 0.2) {
            const trailLen = Math.min(speed * 4, 8);
            const angle = Math.atan2(-agent.vel.y, -agent.vel.x);
            ctx.beginPath();
            ctx.moveTo(px, py);
            ctx.lineTo(px + Math.cos(angle) * trailLen, py + Math.sin(angle) * trailLen);
            ctx.strokeStyle = agent.rerouted
                ? 'rgba(245, 158, 11, 0.3)'
                : 'rgba(59, 130, 246, 0.25)';
            ctx.lineWidth = 2;
            ctx.stroke();
        }

        // Agent circle
        ctx.beginPath();
        ctx.arc(px, py, radius, 0, Math.PI * 2);
        ctx.fillStyle = agent.color || '#3b82f6';
        ctx.fill();

        // Glow for rerouted agents
        if (agent.rerouted) {
            ctx.beginPath();
            ctx.arc(px, py, radius + 2, 0, Math.PI * 2);
            ctx.strokeStyle = 'rgba(245, 158, 11, 0.4)';
            ctx.lineWidth = 1;
            ctx.stroke();
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// RENDERING — Labels Layer
// ═══════════════════════════════════════════════════════════════════

function drawLabelsLayer() {
    if (!simulationState) return;

    ctx.font = '600 8px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    // Bottleneck warning icons
    if (simulationState.hazard_zones) {
        for (const hz of simulationState.hazard_zones) {
            if (hz.severity === 'CRITICAL' || hz.severity === 'EMERGENCY') {
                const px = hz.cell_x * CELL_SIZE + CELL_SIZE / 2;
                const py = hz.cell_y * CELL_SIZE + CELL_SIZE / 2;
                ctx.fillStyle = 'rgba(220, 38, 38, 0.7)';
                ctx.font = '12px sans-serif';
                ctx.fillText('⚠', px, py - 2);
            }
        }
    }

    // Signage arrows
    if (simulationState.signs) {
        const arrowMap = { RIGHT: '→', LEFT: '←', UP: '↑', DOWN: '↓' };
        for (const sign of simulationState.signs) {
            if (sign.active) {
                const px = sign.position.x * CELL_SIZE + CELL_SIZE / 2;
                const py = sign.position.y * CELL_SIZE + CELL_SIZE / 2;
                // Glowing sign background
                ctx.fillStyle = 'rgba(245, 158, 11, 0.2)';
                ctx.fillRect(
                    sign.position.x * CELL_SIZE - 2,
                    sign.position.y * CELL_SIZE - 2,
                    CELL_SIZE + 4, CELL_SIZE + 4
                );
                ctx.fillStyle = '#f59e0b';
                ctx.font = 'bold 14px sans-serif';
                ctx.fillText(arrowMap[sign.direction] || '◆', px, py);
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// RENDERING — Minimap
// ═══════════════════════════════════════════════════════════════════

function drawMinimap() {
    if (!gridConfig) return;

    const mCtx = minimapCtx;
    const scale = 4;

    mCtx.clearRect(0, 0, minimapCanvas.width, minimapCanvas.height);

    // Draw grid
    for (let y = 0; y < gridConfig.height; y++) {
        for (let x = 0; x < gridConfig.width; x++) {
            mCtx.fillStyle = getCellColor(gridConfig.grid[y][x]);
            mCtx.fillRect(x * scale, y * scale, scale, scale);
        }
    }

    // Draw heatmap overlay
    if (heatmapData && heatmapData.heatmap && layers.heatmap) {
        const hm = heatmapData.heatmap;
        for (let y = 0; y < hm.length; y++) {
            for (let x = 0; x < hm[y].length; x++) {
                if (hm[y][x] > 0.3) {
                    mCtx.fillStyle = densityToColor(hm[y][x], 0.7);
                    mCtx.fillRect(x * scale, y * scale, scale, scale);
                }
            }
        }
    }

    // Draw agents as dots
    if (simulationState && simulationState.agents) {
        for (const agent of simulationState.agents) {
            mCtx.fillStyle = agent.rerouted ? '#f59e0b' : '#60a5fa';
            mCtx.fillRect(
                Math.floor(agent.pos.x) * scale + 1,
                Math.floor(agent.pos.y) * scale + 1,
                2, 2
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// MAIN RENDER LOOP
// ═══════════════════════════════════════════════════════════════════

function draw() {
    if (!gridConfig) {
        requestAnimationFrame(draw);
        return;
    }

    // Debug: log once every 300 frames (~5 seconds)
    if (animationFrame % 300 === 0) {
        console.log('[CFO draw]', {
            canvasW: canvas.width, canvasH: canvas.height,
            displayW: canvas.offsetWidth, displayH: canvas.offsetHeight,
            zoomLevel, panX, panY,
            hasBase: !!baseCanvas,
            agents: simulationState ? simulationState.agents.length : 0,
            hasHeatmap: !!(heatmapData && heatmapData.heatmap),
        });
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Apply zoom / pan
    ctx.save();
    ctx.translate(panX, panY);
    ctx.scale(zoomLevel, zoomLevel);

    // Layer 1: Base (from offscreen canvas)
    if (baseCanvas) {
        ctx.drawImage(baseCanvas, 0, 0);
    }

    // Layer 2: Heatmap
    if (layers.heatmap) drawHeatmapLayer();

    // Layer 3: Flow vectors
    if (layers.flow) drawFlowVectors();

    // Layer 4: Reroute paths
    if (layers.paths) drawPathLayer();

    // Layer 5: Agents
    if (layers.agents) drawAgents();

    // Layer 6: Labels & overlays
    if (layers.labels) drawLabelsLayer();
    // Always draw signage and hazard icons on top
    drawSignageOverlay();

    ctx.restore();

    // Minimap (always drawn)
    drawMinimap();

    // Critical alert canvas glow
    if (simulationState && simulationState.hazard_zones) {
        const hasCritical = simulationState.hazard_zones.some(
            hz => hz.severity === 'CRITICAL' || hz.severity === 'EMERGENCY'
        );
        canvas.classList.toggle('critical-alert', hasCritical);
    }

    animationFrame++;
    requestAnimationFrame(draw);
}

function drawSignageOverlay() {
    if (!simulationState || !simulationState.signs) return;
    const arrowMap = { RIGHT: '→', LEFT: '←', UP: '↑', DOWN: '↓', STRAIGHT: '→' };
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    for (const sign of simulationState.signs) {
        if (sign.active) {
            const px = sign.position.x * CELL_SIZE + CELL_SIZE / 2;
            const py = sign.position.y * CELL_SIZE + CELL_SIZE / 2;
            // Pulsing background
            const pulse = 0.15 + 0.1 * Math.sin(animationFrame * 0.08);
            ctx.fillStyle = `rgba(245, 158, 11, ${pulse})`;
            const pad = 4;
            ctx.beginPath();
            ctx.roundRect(
                sign.position.x * CELL_SIZE - pad,
                sign.position.y * CELL_SIZE - pad,
                CELL_SIZE + pad * 2, CELL_SIZE + pad * 2, 4
            );
            ctx.fill();
            ctx.fillStyle = '#fbbf24';
            ctx.font = 'bold 14px sans-serif';
            ctx.fillText(arrowMap[sign.direction] || '◆', px, py);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// UI UPDATES
// ═══════════════════════════════════════════════════════════════════

function updateAllUI() {
    if (!simulationState) return;

    const s = simulationState;

    // Stats
    statAgents.textContent = s.total_agents || s.agents.length;
    statRerouted.textContent = s.total_rerouted || 0;
    statMaxDensity.textContent = (s.max_density || 0).toFixed(1);
    statHazards.textContent = s.hazard_zones ? s.hazard_zones.length : 0;

    // Highlight hazard stat card
    const hasHazards = s.hazard_zones && s.hazard_zones.length > 0;
    document.getElementById('stat-card-bottleneck').classList.toggle('highlight', hasHazards);
    document.getElementById('stat-card-rerouted').classList.toggle('highlight', (s.total_rerouted || 0) > 0);

    // Sim clock
    const totalSec = Math.floor(s.sim_time_sec || 0);
    const hrs = String(Math.floor(totalSec / 3600)).padStart(2, '0');
    const mins = String(Math.floor((totalSec % 3600) / 60)).padStart(2, '0');
    const secs = String(totalSec % 60).padStart(2, '0');
    simClock.textContent = `T + ${hrs}:${mins}:${secs}`;

    // Average density badge
    avgDensityBadge.textContent = `Avg: ${(s.avg_density || 0).toFixed(1)}`;

    // Bottom bar
    document.getElementById('bottom-agents').textContent = s.total_agents || s.agents.length;
    document.getElementById('bottom-sim-time').textContent = `${totalSec}s`;
    document.getElementById('bottom-speed').textContent = `${s.sim_speed || 1}×`;

    // Zone density table
    updateDensityTable(s.zone_densities || []);

    // Gate status
    updateGateList(s.gates || []);

    // Alerts
    updateAlerts(s.alerts || [], s.hazard_zones || []);

    // Prediction timeline
    updatePredictionTimeline(s.predictions || []);

    // Digital signage
    updateSignageList(s.signs || []);

    // Reroute list
    updateRerouteList(s.suggested_routes || {}, s.total_rerouted || 0);
}

function updateDensityTable(zones) {
    const tbody = document.getElementById('density-tbody');
    if (!zones.length) {
        tbody.innerHTML = '<tr><td colspan="4" style="color: var(--text-dim); text-align: center; padding: 12px;">No density data</td></tr>';
        return;
    }

    // Sort by density descending, show top 15
    const sorted = [...zones].sort((a, b) => b.density - a.density).slice(0, 15);

    tbody.innerHTML = sorted.map(z => {
        const trendIcon = z.trend === 'rising' ? '▲' : z.trend === 'falling' ? '▼' : '—';
        const trendClass = z.trend;
        return `<tr>
            <td>(${z.cell_x},${z.cell_y})</td>
            <td>${z.density.toFixed(1)}</td>
            <td><span class="los-badge ${z.los_level}">${z.los_level}</span></td>
            <td><span class="trend-indicator ${trendClass}">${trendIcon}</span></td>
        </tr>`;
    }).join('');
}

function updateGateList(gates) {
    const container = document.getElementById('gate-list');
    if (!gates.length) return;

    container.innerHTML = gates.map(g => {
        const dotClass = g.status.includes('THROTTL') || g.status.includes('RESTRICT')
            ? 'throttled'
            : g.action === 'CLOSE' ? 'closed' : 'open';
        return `<div class="gate-item">
            <div class="gate-info">
                <span class="gate-dot ${dotClass}"></span>
                <span>${g.gate_id}</span>
            </div>
            <span class="gate-rate">${g.target_rate_per_sec.toFixed(1)}/s</span>
            <div class="gate-controls">
                <button class="gate-btn open-btn" onclick="controlGate('${g.gate_id}','OPEN_FULL')" title="Open">●</button>
                <button class="gate-btn throttle-btn" onclick="controlGate('${g.gate_id}','THROTTLE_FLOW')" title="Throttle">◐</button>
                <button class="gate-btn close-btn" onclick="controlGate('${g.gate_id}','CLOSE')" title="Close">○</button>
            </div>
        </div>`;
    }).join('');
}

function updateAlerts(alerts, hazards) {
    if (!alerts.length && (!hazards || hazards.length === 0)) {
        alertList.innerHTML = '<li class="alert-item empty-alert">System nominal. No active hazards detected.</li>';
        document.getElementById('alert-count').textContent = '0';
        return;
    }

    // Combine hazard-based alerts
    const items = [];

    if (alerts.length > 0) {
        for (const a of alerts.slice(0, 15)) {
            items.push(`<li class="alert-item ${a.severity}">
                <span>${a.message}</span>
                <span class="alert-time">${new Date(a.timestamp * 1000).toLocaleTimeString()}</span>
            </li>`);
        }
    } else if (hazards.length > 0) {
        // Fallback: generate from hazards directly
        const sevCounts = { EMERGENCY: 0, CRITICAL: 0, WARNING: 0, INFO: 0 };
        for (const hz of hazards) {
            sevCounts[hz.severity] = (sevCounts[hz.severity] || 0) + 1;
        }
        if (sevCounts.EMERGENCY > 0) {
            items.push(`<li class="alert-item EMERGENCY">🚨 EMERGENCY: ${sevCounts.EMERGENCY} zones at crush-risk density</li>`);
        }
        if (sevCounts.CRITICAL > 0) {
            items.push(`<li class="alert-item CRITICAL">🔴 ${sevCounts.CRITICAL} critical congestion zones — rerouting active</li>`);
        }
        if (sevCounts.WARNING > 0) {
            items.push(`<li class="alert-item WARNING">⚠ ${sevCounts.WARNING} zones with rising density</li>`);
        }
    }

    alertList.innerHTML = items.join('') || '<li class="alert-item empty-alert">System nominal. No active hazards detected.</li>';
    document.getElementById('alert-count').textContent = String(items.length);
}

function updatePredictionTimeline(predictions) {
    const container = document.getElementById('prediction-timeline');
    const statusBadge = document.getElementById('prediction-status');

    if (!predictions.length) {
        container.innerHTML = '<div style="font-size: 0.72rem; color: var(--text-dim); text-align: center; padding: 16px;">Predictions generate after 5+ agents are active</div>';
        statusBadge.textContent = 'Idle';
        return;
    }

    statusBadge.textContent = 'Active';
    const maxPossibleDensity = 5.0;

    container.innerHTML = predictions.map(p => {
        const mins = Math.floor(p.timestamp_offset_sec / 60);
        const secs = Math.floor(p.timestamp_offset_sec % 60);
        const timeStr = `+${mins}:${String(secs).padStart(2, '0')}`;
        const fillPct = Math.min(100, (p.max_density / maxPossibleDensity) * 100);
        const color = densityToColor(p.max_density, 1.0);
        const densityColor = losToColor(
            p.max_density > 3.5 ? 'CRITICAL' :
            p.max_density > 2.17 ? 'F' :
            p.max_density > 1.54 ? 'E' :
            p.max_density > 1.08 ? 'D' : 'C'
        );

        return `<div class="prediction-item">
            <span class="prediction-time">${timeStr}</span>
            <div class="prediction-bar">
                <div class="prediction-fill" style="width: ${fillPct}%; background: ${color};"></div>
            </div>
            <span class="prediction-density" style="color: ${densityColor}">${p.max_density.toFixed(1)}</span>
        </div>`;
    }).join('');
}

function updateSignageList(signs) {
    const container = document.getElementById('sign-list');
    if (!signs.length) return;

    const arrowMap = { RIGHT: '→', LEFT: '←', UP: '↑', DOWN: '↓', STRAIGHT: '→' };
    container.innerHTML = signs.map(s => {
        return `<div class="sign-item ${s.active ? 'active' : ''}">
            <span class="sign-direction">${arrowMap[s.direction] || '◆'}</span>
            <span class="sign-message">${s.message || 'Normal flow'}</span>
        </div>`;
    }).join('');
}

function updateRerouteList(routes, count) {
    const container = document.getElementById('reroute-list');
    document.getElementById('reroute-count').textContent = String(count);

    const routeKeys = Object.keys(routes).slice(0, 8);
    if (!routeKeys.length) {
        container.innerHTML = '<div style="font-size: 0.72rem; color: var(--text-dim); text-align: center; padding: 12px;">No active reroutes</div>';
        return;
    }

    container.innerHTML = routeKeys.map(id => {
        const path = routes[id];
        const len = path ? path.length : 0;
        const start = path && path[0] ? `(${Math.round(path[0].x)},${Math.round(path[0].y)})` : '?';
        const end = path && path[len - 1] ? `(${Math.round(path[len - 1].x)},${Math.round(path[len - 1].y)})` : '?';
        return `<div style="font-size: 0.72rem; padding: 5px 8px; background: var(--bg-card); border-radius: 4px; margin-bottom: 4px; display: flex; justify-content: space-between;">
            <span style="color: var(--severity-warning);">Agent #${id}</span>
            <span style="color: var(--text-tertiary);">${start} → ${end}</span>
        </div>`;
    }).join('');
}

// ═══════════════════════════════════════════════════════════════════
// GATE CONTROL
// ═══════════════════════════════════════════════════════════════════

window.controlGate = function (gateId, action) {
    sendCommand({ action: 'gate_control', gate_id: gateId, gate_action: action, rate: action === 'THROTTLE_FLOW' ? 1.0 : 3.5 });
};

// ═══════════════════════════════════════════════════════════════════
// CONTROLS
// ═══════════════════════════════════════════════════════════════════

crowdSizeInput.addEventListener('input', (e) => {
    crowdSizeVal.textContent = e.target.value;
});

btnSpawn.addEventListener('click', () => {
    sendCommand({ action: 'spawn', count: parseInt(crowdSizeInput.value, 10) });
});

btnClear.addEventListener('click', () => {
    sendCommand({ action: 'clear' });
    prevAgentPositions = {};
});

// Speed controls
document.querySelectorAll('.speed-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.speed-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const speed = parseFloat(btn.dataset.speed);
        sendCommand({ action: 'set_speed', speed });
    });
});

// Layer toggles
document.querySelectorAll('.overlay-toggle').forEach(btn => {
    btn.addEventListener('click', () => {
        const layer = btn.dataset.layer;
        layers[layer] = !layers[layer];
        btn.classList.toggle('active', layers[layer]);
    });
});

// ═══════════════════════════════════════════════════════════════════
// ZOOM & PAN
// ═══════════════════════════════════════════════════════════════════

canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    const zoomDelta = e.deltaY > 0 ? 0.9 : 1.1;
    const newZoom = Math.max(0.5, Math.min(3.0, zoomLevel * zoomDelta));

    // Zoom toward cursor
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    panX = mx - (mx - panX) * (newZoom / zoomLevel);
    panY = my - (my - panY) * (newZoom / zoomLevel);
    zoomLevel = newZoom;
}, { passive: false });

canvas.addEventListener('mousedown', (e) => {
    if (e.button === 0) {
        isDragging = true;
        dragStartX = e.clientX - panX;
        dragStartY = e.clientY - panY;
        canvas.style.cursor = 'grabbing';
    }
});

canvas.addEventListener('mousemove', (e) => {
    if (isDragging) {
        panX = e.clientX - dragStartX;
        panY = e.clientY - dragStartY;
    }

    // Tooltip: show density info on hover
    if (gridConfig && heatmapData && heatmapData.heatmap) {
        const rect = canvas.getBoundingClientRect();
        const mx = (e.clientX - rect.left - panX) / zoomLevel;
        const my = (e.clientY - rect.top - panY) / zoomLevel;
        const cellX = Math.floor(mx / CELL_SIZE);
        const cellY = Math.floor(my / CELL_SIZE);

        if (cellX >= 0 && cellX < gridConfig.width && cellY >= 0 && cellY < gridConfig.height) {
            const density = heatmapData.heatmap[cellY] ? heatmapData.heatmap[cellY][cellX] || 0 : 0;
            const cellType = gridConfig.grid[cellY][cellX];

            if (density > 0.1 || cellType !== 'empty') {
                const los = density > 3.5 ? 'CRIT' : density > 2.17 ? 'F' : density > 1.54 ? 'E' : density > 1.08 ? 'D' : density > 0.43 ? 'C' : density > 0.31 ? 'B' : 'A';
                tooltip.innerHTML = `
                    <div class="tooltip-row"><span class="tooltip-label">Cell</span><span class="tooltip-value">(${cellX}, ${cellY})</span></div>
                    <div class="tooltip-row"><span class="tooltip-label">Type</span><span class="tooltip-value">${cellType}</span></div>
                    <div class="tooltip-row"><span class="tooltip-label">Density</span><span class="tooltip-value">${density.toFixed(2)} p/m²</span></div>
                    <div class="tooltip-row"><span class="tooltip-label">LoS</span><span class="tooltip-value" style="color:${losToColor(los)}">${los}</span></div>
                `;
                tooltip.style.left = (e.clientX + 12) + 'px';
                tooltip.style.top = (e.clientY + 12) + 'px';
                tooltip.classList.add('visible');
            } else {
                tooltip.classList.remove('visible');
            }
        } else {
            tooltip.classList.remove('visible');
        }
    }
});

canvas.addEventListener('mouseup', () => {
    isDragging = false;
    canvas.style.cursor = 'grab';
});

canvas.addEventListener('mouseleave', () => {
    isDragging = false;
    canvas.style.cursor = 'grab';
    tooltip.classList.remove('visible');
});

// ═══════════════════════════════════════════════════════════════════
// INIT
// ═══════════════════════════════════════════════════════════════════

initWebSocket();
requestAnimationFrame(draw);

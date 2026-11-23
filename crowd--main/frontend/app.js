/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   CROWD FLOW OPTIMISER — Multi-Layer Canvas Rendering Engine
   Features: Heatmap, Flow Vectors, Path Animations, Interpolation,
             Minimap, Tooltips, Zoom/Pan, Event Setup, Recommended Actions
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */

// ═══════════════════════════════════════════════════════════════════
// STATE
// ═══════════════════════════════════════════════════════════════════

let ws;
let gridConfig = null;
let simulationState = null;
let heatmapData = null;
let visionData = null;
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

// Manual spawn controls (may be absent -> guarded below)
const btnSpawn = document.getElementById('btn-spawn');
const btnClear = document.getElementById('btn-clear');
const crowdSizeInput = document.getElementById('crowd-size');
const crowdSizeVal = document.getElementById('crowd-size-val');

const btnVisionStart = document.getElementById('btn-vision-start');
const btnVisionStop = document.getElementById('btn-vision-stop');
const visionStatus = document.getElementById('vision-status');

const statAgents = document.getElementById('stat-agents');
const statRerouted = document.getElementById('stat-rerouted');
const statMaxDensity = document.getElementById('stat-max-density');
const statHazards = document.getElementById('stat-hazards');
const alertList = document.getElementById('alert-list');
const simClock = document.getElementById('sim-clock');       // top bar clock (T + hh:mm:ss)
const tickCounter = document.getElementById('tick-counter');
const avgDensityBadge = document.getElementById('avg-density');
const connectionDot = document.getElementById('connection-dot');
const connectionText = document.getElementById('connection-text');

// ═══════════════════════════════════════════════════════════════════
// COLOR UTILITIES
// ═══════════════════════════════════════════════════════════════════

function densityToColor(d, alpha = 1.0) {
    const lerp = (a, b, t) => Math.floor(a + (b - a) * t);

    if (d < 0.3) return `rgba(18, 51, 58, ${alpha * 0.35})`;

    if (d < 1.0) {
        const t = (d - 0.3) / 0.7;
        return `rgba(${lerp(18, 111, t)}, ${lerp(51, 154, t)}, ${lerp(58, 126, t)}, ${alpha * (0.35 + t * 0.15)})`;
    }
    if (d < 2.0) {
        const t = (d - 1.0) / 1.0;
        return `rgba(${lerp(111, 244, t)}, ${lerp(154, 213, t)}, ${lerp(126, 141, t)}, ${alpha * (0.50 + t * 0.15)})`;
    }
    if (d < 3.5) {
        const t = (d - 2.0) / 1.5;
        return `rgba(${lerp(244, 191, t)}, ${lerp(213, 35, t)}, ${lerp(141, 4, t)}, ${alpha * (0.65 + t * 0.20)})`;
    }
    return `rgba(141, 8, 1, ${alpha * 0.92})`;
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
            drawBaseLayer();
        } else if (data.type === 'state_update') {
            if (simulationState && simulationState.agents) {
                for (const a of simulationState.agents) {
                    prevAgentPositions[a.id] = { x: a.pos.x, y: a.pos.y };
                }
            }
            simulationState = data.state;
            heatmapData = data.heatmap;
            updateAllUI();
        } else if (data.type === 'vision_update') {
            visionData = data.data;
            updateVisionDashboard(visionData);
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

            bCtx.strokeStyle = 'rgba(148, 163, 184, 0.04)';
            bCtx.lineWidth = 0.5;
            bCtx.strokeRect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE);
        }
    }

    for (let y = 0; y < gridConfig.height; y++) {
        for (let x = 0; x < gridConfig.width; x++) {
            const cell = gridConfig.grid[y][x];
            if (cell === 'gate' || cell === 'exit' || cell === 'concession') {
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

        const alpha = Math.min(0.7, 0.2 + v.mag * 0.4);
        ctx.strokeStyle = `rgba(96, 165, 250, ${alpha})`;
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(ex, ey);
        ctx.stroke();

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
        let renderX = agent.pos.x;
        let renderY = agent.pos.y;

        const prev = prevAgentPositions[agent.id];
        if (prev) {
            renderX = prev.x + (agent.pos.x - prev.x) * INTERP_FACTOR;
            renderY = prev.y + (agent.pos.y - prev.y) * INTERP_FACTOR;
            prevAgentPositions[agent.id] = { x: renderX, y: renderY };
        }

        const px = renderX * CELL_SIZE + CELL_SIZE / 2;
        const py = renderY * CELL_SIZE + CELL_SIZE / 2;
        const radius = CELL_SIZE / 2.8;

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

        ctx.beginPath();
        ctx.arc(px, py, radius, 0, Math.PI * 2);
        ctx.fillStyle = agent.color || '#3b82f6';
        ctx.fill();

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
}

// ═══════════════════════════════════════════════════════════════════
// RENDERING — Minimap
// ═══════════════════════════════════════════════════════════════════

function drawMinimap() {
    if (!gridConfig) return;

    const mCtx = minimapCtx;
    const scale = 4;

    mCtx.clearRect(0, 0, minimapCanvas.width, minimapCanvas.height);

    for (let y = 0; y < gridConfig.height; y++) {
        for (let x = 0; x < gridConfig.width; x++) {
            mCtx.fillStyle = getCellColor(gridConfig.grid[y][x]);
            mCtx.fillRect(x * scale, y * scale, scale, scale);
        }
    }

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

    if (simulationState && simulationState.real_agents) {
        for (const real_agent of simulationState.real_agents) {
            mCtx.fillStyle = '#ec4899';
            mCtx.beginPath();
            mCtx.arc(
                Math.floor(real_agent.x) * scale + 2,
                Math.floor(real_agent.y) * scale + 2,
                3, 0, Math.PI * 2
            );
            mCtx.fill();
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

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    ctx.save();
    ctx.translate(panX, panY);
    ctx.scale(zoomLevel, zoomLevel);

    if (baseCanvas) ctx.drawImage(baseCanvas, 0, 0);
    if (layers.heatmap) drawHeatmapLayer();
    if (layers.flow) drawFlowVectors();
    if (layers.paths) drawPathLayer();
    if (layers.agents) drawAgents();
    if (layers.labels) drawLabelsLayer();
    drawSignageOverlay();

    ctx.restore();

    drawMinimap();

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

    const hasHazards = s.hazard_zones && s.hazard_zones.length > 0;
    document.getElementById('stat-card-bottleneck').classList.toggle('highlight', hasHazards);
    document.getElementById('stat-card-rerouted').classList.toggle('highlight', (s.total_rerouted || 0) > 0);

    // Sim clock (top bar)
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

    updateDensityTable(s.zone_densities || []);
    updateGateList(s.gates || []);
    updateAlerts(s.alerts || [], s.hazard_zones || []);
    updatePredictionTimeline(s.predictions || []);
    updateSignageList(s.signs || []);
    updateRerouteList(s.suggested_routes || {}, s.total_rerouted || 0);

    // NEW: Recommended Actions + Event Setup clock
    renderRecommendedActions(s);

    const esClock = document.getElementById('es-clock');
    if (esClock) esClock.textContent = hourLabel(s.clock_hour ?? 0);
}

function updateDensityTable(zones) {
    const tbody = document.getElementById('density-tbody');
    if (!zones.length) {
        tbody.innerHTML = '<tr><td colspan="4" style="color: var(--text-dim); text-align: center; padding: 12px;">No density data</td></tr>';
        return;
    }

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
        const status = String(g.status || '');
        const dotClass = status.includes('THROTTL') || status.includes('RESTRICT')
            ? 'throttled'
            : g.action === 'CLOSE' ? 'closed' : 'open';
        const q = g.queue_length ?? 0;
        const wait = g.wait_time_sec ?? 0;
        const waitLabel = q === 0 ? '' : (g.action === 'CLOSE' ? 'closed' : formatWait(wait));
        return `<div class="gate-item">
            <div class="gate-info">
                <span class="gate-dot ${dotClass}"></span>
                <span>${g.gate_id}</span>
            </div>
            <span class="gate-queue ${q > 0 ? 'has-queue' : ''}" title="People waiting outside / estimated wait">${q} q${waitLabel ? ' · ' + waitLabel : ''}</span>
            <span class="gate-rate">${(g.target_rate_per_sec ?? 0).toFixed(1)}/s</span>
            <div class="gate-controls">
                <button class="gate-btn open-btn" onclick="controlGate('${g.gate_id}','OPEN_FULL')" title="Open">●</button>
                <button class="gate-btn throttle-btn" onclick="controlGate('${g.gate_id}','THROTTLE_FLOW')" title="Throttle">◐</button>
                <button class="gate-btn close-btn" onclick="controlGate('${g.gate_id}','CLOSE')" title="Close">○</button>
            </div>
        </div>`;
    }).join('');
}

function formatWait(sec) {
    if (sec < 60) return `${Math.round(sec)}s`;
    return `${Math.floor(sec / 60)}m ${String(Math.round(sec % 60)).padStart(2, '0')}s`;
}

function updateAlerts(alerts, hazards) {
    if (!alerts.length && (!hazards || hazards.length === 0)) {
        alertList.innerHTML = '<li class="alert-item empty-alert">System nominal. No active hazards detected.</li>';
        document.getElementById('alert-count').textContent = '0';
        return;
    }

    const items = [];

    if (alerts.length > 0) {
        for (const a of alerts.slice(0, 15)) {
            items.push(`<li class="alert-item ${a.severity}">
                <span>${a.message}</span>
                <span class="alert-time">${new Date(a.timestamp * 1000).toLocaleTimeString()}</span>
            </li>`);
        }
    } else if (hazards.length > 0) {
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
// RECOMMENDED ACTIONS (computed client-side from hazard_zones + gates)
// Heuristic: nearest open gate by rate, sorted by time-to-choke.
// Not a queueing model.
// ═══════════════════════════════════════════════════════════════════

// LoS-C safe density ceiling (people/m²) — matches FRUIN_THRESHOLDS in
// backend/density.py. Used to compute a real "divert %": how much inflow
// has to drop for the worst zone to fall back under this line.
const SAFE_DENSITY_PM2 = 1.08;

// Cached list of named landmarks built once from the static grid
// (gridConfig.grid), not invented — every entry corresponds to an actual
// 'concession' cell the backend placed on the venue layout.
let _landmarkZonesCache = null;

// key: "originGateId>targetGateId" -> { pct, until }
// Cards rebuild via innerHTML every tick, so a button's own state can't
// persist — this is what lets the card show "✓ Diverted" for a few seconds
// after a click instead of silently reverting to the same button.
const _recentDiverts = new Map();
const DIVERT_CONFIRM_MS = 4000;

function getLandmarkZones() {
    if (_landmarkZonesCache) return _landmarkZonesCache;
    if (!gridConfig || !gridConfig.grid) return [];

    const landmarks = [];
    let foodCourtCount = 0;
    for (let y = 0; y < gridConfig.height; y++) {
        for (let x = 0; x < gridConfig.width; x++) {
            if (gridConfig.grid[y][x] === 'concession') {
                foodCourtCount++;
                landmarks.push({ x, y, label: `Food Court ${foodCourtCount}` });
            }
        }
    }
    _landmarkZonesCache = landmarks;
    return landmarks;
}

// Names a hazard by the nearest known landmark within range, otherwise
// falls back to the raw cell coordinates (same as before).
function zoneLabel(cellX, cellY) {
    const landmarks = getLandmarkZones();
    let nearest = null;
    let nearestDist = Infinity;
    for (const lm of landmarks) {
        const d = Math.hypot(lm.x - cellX, lm.y - cellY);
        if (d < nearestDist) {
            nearestDist = d;
            nearest = lm;
        }
    }
    if (nearest && nearestDist <= 6) return nearest.label;
    return `Zone (${cellX}, ${cellY})`;
}

// Nearest gate (any status) to a given cell — treated as the "origin"
// gate feeding that congested zone, using each gate's real position.x/y.
function nearestGate(gates, cellX, cellY) {
    let best = null;
    let bestDist = Infinity;
    for (const g of gates) {
        const gx = g.position ? g.position.x : null;
        const gy = g.position ? g.position.y : null;
        if (gx == null || gy == null) continue;
        const d = Math.hypot(gx - cellX, gy - cellY);
        if (d < bestDist) {
            bestDist = d;
            best = g;
        }
    }
    return best;
}

function gateLabel(g) {
    return g ? String(g.gate_id).replace('gate_', 'Gate ') : null;
}

// The panel used to rebuild on every state_update — at 2x-10x speed that's
// 4 to 10 full DOM rebuilds per second, which a human can't read and which
// destroys/recreates the divert buttons before anyone can click them. This
// throttles actual re-renders to roughly human reading speed, independent
// of sim speed. handleDivert() resets the timer to 0 so a click's own
// confirmation state renders on the very next tick instead of waiting out
// the rest of the window.
let _raLastRender = 0;
const RA_MIN_INTERVAL_MS = 1500;

function renderRecommendedActions(s) {
    if (Date.now() - _raLastRender < RA_MIN_INTERVAL_MS) return;
    _raLastRender = Date.now();
    _renderRecommendedActionsNow(s);
}

function _renderRecommendedActionsNow(s) {
    const list = document.getElementById('ra-list');
    const count = document.getElementById('ra-count');
    if (!list || !count) return;

    const hazards = (s.hazard_zones || [])
        .filter(z => ['WARNING', 'CRITICAL', 'EMERGENCY'].includes(z.severity))
        // time_to_choke_sec first; cell_x/cell_y as a deterministic tiebreak
        // so zones tied on danger stop reshuffling every tick — several
        // zones landing on the exact same density (a common plateau) used
        // to reorder randomly each rebuild because only the primary key
        // was compared.
        .sort((a, b) =>
            (a.time_to_choke_sec ?? 9999) - (b.time_to_choke_sec ?? 9999) ||
            (a.cell_x - b.cell_x) ||
            (a.cell_y - b.cell_y))
        .slice(0, 4);

    count.textContent = hazards.length;
    count.classList.toggle('zero', hazards.length === 0);

    if (!hazards.length) {
        list.innerHTML = '<div class="ra-empty">No action needed. All zones within safe limits.</div>';
        return;
    }

    const allGates = s.gates || [];
    // gates jo closed/throttled nahi hain
    const openGates = allGates.filter(g => {
        const status = String(g.status || '');
        return g.action !== 'CLOSE' &&
            !status.includes('THROTTL') &&
            !status.includes('RESTRICT');
    });

    const cards = hazards.map(z => {
        const eta = z.time_to_choke_sec;
        // time_to_choke_sec is 0.0 when density is not rising (no estimate)
        const etaLabel = (eta != null && eta > 0 && eta < 600) ? `${Math.max(1, Math.round(eta / 60))} min` : '';
        const dens = Number(z.current_density ?? z.density ?? 0);
        const name = zoneLabel(z.cell_x, z.cell_y);

        // Per-zone origin gate: whichever real gate sits closest to this
        // hazard's cell — that's the one presumed to be feeding it.
        const originGate = nearestGate(allGates, z.cell_x, z.cell_y);

        // Least-loaded open gate that ISN'T the origin — shortest queue,
        // then highest admit rate.
        const target = [...openGates]
            .filter(g => !originGate || g.gate_id !== originGate.gate_id)
            .sort((a, b) =>
                ((a.queue_length ?? 0) - (b.queue_length ?? 0)) ||
                ((b.target_rate_per_sec ?? 0) - (a.target_rate_per_sec ?? 0)))[0];

        // Per-zone divert %: how much inflow has to drop for THIS zone to
        // fall back under the LoS-C safe ceiling (1.08 p/m², same threshold
        // backend/density.py uses). Real formula, not a flat guess.
        let pct = 0;
        if (dens > SAFE_DENSITY_PM2) {
            pct = Math.round(((dens - SAFE_DENSITY_PM2) / dens) * 100);
            pct = Math.max(5, Math.min(95, pct));
        }

        const originLabel = gateLabel(originGate);
        const targetLabel = gateLabel(target);
        const etaPhrase = etaLabel ? ` in ${etaLabel}` : '';

        let divertBlock = '';
        let action;
        if (target && originGate && originLabel && targetLabel && originLabel !== targetLabel && pct > 0) {
            // Every number here is a real field off the gate objects — nothing invented.
            const oq = originGate.queue_length ?? 0;
            const ow = originGate.wait_time_sec ?? 0;
            const tq = target.queue_length ?? 0;
            const tRate = target.target_rate_per_sec ?? 0;

            action = `${originLabel}: ${oq} queuing, ${ow.toFixed(1)}s wait. `
                    + `Divert ${pct}% of new arrivals to ${targetLabel} — `
                    + `currently ${tq} queuing, admitting ${tRate.toFixed(1)}/s.`;

            const key = `${originGate.gate_id}>${target.gate_id}`;
            const recent = _recentDiverts.get(key);

            divertBlock = recent
                ? `<div class="ra-divert-confirm">✓ Diverted ${recent.pct}% — draining ${originLabel}</div>`
                : `<div class="ra-divert-row">
                     <span class="ra-divert-label">${originLabel} → ${targetLabel}</span>
                     <button class="ra-divert-btn"
                             data-origin="${originGate.gate_id}"
                             data-target="${target.gate_id}"
                             data-pct="${pct}">
                       Divert ${pct}%
                     </button>
                   </div>`;
        } else if (target && targetLabel) {
            action = `${name} reaches LoS ${z.fruin_los || z.severity} (${dens.toFixed(1)} p/m²)${etaPhrase}. Shift intake toward <b>${targetLabel}</b>.`;
        } else {
            action = `${name} reaches LoS ${z.fruin_los || z.severity} (${dens.toFixed(1)} p/m²)${etaPhrase}. No clearer alternative — throttle intake.`;
        }

        return `
          <div class="ra-card ${z.severity === 'WARNING' ? 'warn' : ''}">
            <div class="ra-top">
              <span class="ra-title">${name} congested</span>
              ${etaLabel ? `<span class="ra-eta">${etaLabel}</span>` : ''}
            </div>
            <div class="ra-meta">${dens.toFixed(1)} p/m² · LoS ${z.fruin_los || z.severity}</div>
            ${divertBlock}
            <div class="ra-body">${action}</div>
            ${!target ? '<div class="ra-nospare">no alternative with spare capacity</div>' : ''}
          </div>`;
    }).join('');

    list.innerHTML = cards;
}

function handleDivert(originGateId, targetGateId, pct) {
    sendCommand({
        action: 'divert_gate',
        origin_gate_id: originGateId,
        target_gate_id: targetGateId,
        pct: pct,
    });

    const key = `${originGateId}>${targetGateId}`;
    _recentDiverts.set(key, { pct, until: Date.now() + DIVERT_CONFIRM_MS });
    // Bypass the throttle once, so the confirmation shows on the very next
    // state_update instead of waiting out whatever's left of the window.
    _raLastRender = 0;
    setTimeout(() => {
        const entry = _recentDiverts.get(key);
        if (entry && Date.now() >= entry.until) _recentDiverts.delete(key);
    }, DIVERT_CONFIRM_MS + 100);
}

// One delegated listener for the whole panel — cards are replaced wholesale
// every tick, so listeners attached to individual buttons would leak/detach.
document.getElementById('ra-list')?.addEventListener('click', (e) => {
    const btn = e.target.closest('.ra-divert-btn');
    if (!btn) return;
    handleDivert(btn.dataset.origin, btn.dataset.target, Number(btn.dataset.pct));
});

// ═══════════════════════════════════════════════════════════════════
// GATE CONTROL
// ═══════════════════════════════════════════════════════════════════

window.controlGate = function (gateId, action) {
    sendCommand({ action: 'gate_control', gate_id: gateId, gate_action: action, rate: action === 'THROTTLE_FLOW' ? 1.0 : 3.5 });
};

// ═══════════════════════════════════════════════════════════════════
// CONTROLS (guarded: panel missing ho to crash nahi hoga)
// ═══════════════════════════════════════════════════════════════════

if (crowdSizeInput && crowdSizeVal) {
    crowdSizeInput.addEventListener('input', (e) => {
        crowdSizeVal.textContent = e.target.value;
    });
}

if (btnSpawn && crowdSizeInput) {
    btnSpawn.addEventListener('click', () => {
        sendCommand({ action: 'spawn', count: parseInt(crowdSizeInput.value, 10) });
    });
}

if (btnClear) {
    btnClear.addEventListener('click', () => {
        sendCommand({ action: 'clear' });
        prevAgentPositions = {};
    });
}

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
// VISION PIPELINE CONTROLS
// ═══════════════════════════════════════════════════════════════════

btnVisionStart.addEventListener('click', () => {
    fetch('/api/vision/start', { method: 'POST' })
        .then(r => r.json())
        .then(() => {
            visionStatus.textContent = 'Live';
            visionStatus.style.backgroundColor = '#10b981';
            visionStatus.style.color = '#fff';
        })
        .catch(err => console.error(err));
});

btnVisionStop.addEventListener('click', () => {
    fetch('/api/vision/stop', { method: 'POST' })
        .then(r => r.json())
        .then(() => {
            visionStatus.textContent = 'Stopped';
            visionStatus.style.backgroundColor = '';
            visionStatus.style.color = '';
        })
        .catch(err => console.error(err));
});

// ═══════════════════════════════════════════════════════════════════
// VIDEO FILE PICKER & SCAN
// ═══════════════════════════════════════════════════════════════════

let selectedVideoPaths = [];

function updateFileListUI() {
    const listEl = document.getElementById('video-file-list');
    if (!listEl) return;
    if (selectedVideoPaths.length === 0) {
        listEl.innerHTML = '<span style="color:var(--text-dim);">No files selected</span>';
        return;
    }
    let html = '';
    selectedVideoPaths.forEach((p, i) => {
        const name = p.split(/[\\/]/).pop();
        const isFolder = !name.includes('.');
        const icon = isFolder ? '📂' : '🎬';
        html += `<div style="display:flex;justify-content:space-between;align-items:center;padding:2px 0;border-bottom:1px solid rgba(255,255,255,0.04);">`;
        html += `<span style="color:var(--text-primary);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:200px;" title="${p}">${icon} ${name}</span>`;
        html += `<span style="color:var(--text-dim);cursor:pointer;padding:0 4px;" onclick="removeVideoPath(${i})">✕</span>`;
        html += `</div>`;
    });
    listEl.innerHTML = html;
}

window.removeVideoPath = function (index) {
    selectedVideoPaths.splice(index, 1);
    updateFileListUI();
};

const btnPickFiles = document.getElementById('btn-pick-files');
if (btnPickFiles) {
    btnPickFiles.addEventListener('click', () => {
        btnPickFiles.textContent = '⏳ Opening...';
        fetch('/api/vision/pick-files')
            .then(r => r.json())
            .then(data => {
                if (data.paths && data.paths.length > 0) {
                    data.paths.forEach(p => {
                        if (!selectedVideoPaths.includes(p)) selectedVideoPaths.push(p);
                    });
                    updateFileListUI();
                }
                btnPickFiles.textContent = '📁 Add Files';
            })
            .catch(err => {
                console.error(err);
                btnPickFiles.textContent = '📁 Add Files';
            });
    });
}

const btnPickFolder = document.getElementById('btn-pick-folder');
if (btnPickFolder) {
    btnPickFolder.addEventListener('click', () => {
        btnPickFolder.textContent = '⏳ Opening...';
        fetch('/api/vision/pick-folder')
            .then(r => r.json())
            .then(data => {
                if (data.path) {
                    if (!selectedVideoPaths.includes(data.path)) selectedVideoPaths.push(data.path);
                    updateFileListUI();
                }
                btnPickFolder.textContent = '📂 Add Folder';
            })
            .catch(err => {
                console.error(err);
                btnPickFolder.textContent = '📂 Add Folder';
            });
    });
}

const btnClearFiles = document.getElementById('btn-clear-files');
if (btnClearFiles) {
    btnClearFiles.addEventListener('click', () => {
        selectedVideoPaths = [];
        updateFileListUI();
        const pathInput = document.getElementById('video-path-input');
        if (pathInput) pathInput.value = '';
    });
}

const btnScanVideo = document.getElementById('btn-scan-video');
const videoPathInput = document.getElementById('video-path-input');
if (btnScanVideo) {
    btnScanVideo.addEventListener('click', () => {
        let allPaths = [...selectedVideoPaths];
        if (videoPathInput && videoPathInput.value.trim()) {
            const manual = videoPathInput.value.trim().split(',').map(p => p.trim()).filter(Boolean);
            manual.forEach(p => {
                if (!allPaths.includes(p)) allPaths.push(p);
            });
        }

        if (allPaths.length === 0) {
            alert('Please select video files or folders first.');
            return;
        }

        btnScanVideo.textContent = '⏳ Loading...';
        fetch('/api/vision/scan-video', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ paths: allPaths }),
        })
            .then(r => r.json())
            .then(() => {
                visionStatus.textContent = 'Scanning';
                visionStatus.style.backgroundColor = '#f59e0b';
                visionStatus.style.color = '#000';
                btnScanVideo.textContent = '▶ Scan';
            })
            .catch(err => {
                console.error(err);
                btnScanVideo.textContent = '▶ Scan';
            });
    });
}

// ═══════════════════════════════════════════════════════════════════
// VISION DASHBOARD UPDATE
// ═══════════════════════════════════════════════════════════════════

function updateVisionDashboard(data) {
    if (!data) return;

    const fpsEl = document.getElementById('vision-fps');
    const latencyEl = document.getElementById('vision-latency');
    const modeEl = document.getElementById('vision-mode');
    if (fpsEl) fpsEl.textContent = data.fps || '0';
    if (latencyEl) latencyEl.textContent = (data.latency_ms || 0) + 'ms';
    if (modeEl) modeEl.textContent = data.mode || 'idle';

    const detEl = document.getElementById('vision-detections');
    if (detEl && data.detections) {
        let html = '';
        for (const [task, count] of Object.entries(data.detections)) {
            const colorMap = { crowd: '#3b82f6', traffic: '#f59e0b', anomaly: '#ef4444', flow: '#8b5cf6' };
            const color = colorMap[task] || '#64748b';
            html += `<div class="vision-det-row">`;
            html += `<span class="vision-det-label" style="color:${color}">● ${task}</span>`;
            html += `<span class="vision-det-count">${count}</span></div>`;
        }
        detEl.innerHTML = html || '<div style="color:var(--text-dim);font-size:0.72rem;">No detections</div>';
    }

    const progressEl = document.getElementById('vision-progress');
    if (progressEl && data.total_frames > 0) {
        const pct = Math.round((data.frame / data.total_frames) * 100);
        progressEl.style.width = pct + '%';
        progressEl.parentElement.style.display = 'block';
    } else if (progressEl) {
        progressEl.parentElement.style.display = data.mode === 'video' || data.mode === 'batch' ? 'block' : 'none';
    }

    const vlmEl = document.getElementById('vlm-analysis');
    if (vlmEl && data.vlm_analysis) {
        const a = data.vlm_analysis;
        const riskColors = { none: '#22c55e', low: '#4ade80', moderate: '#f59e0b', high: '#ef4444' };
        const densityColors = { low: '#22c55e', moderate: '#f59e0b', high: '#f97316', critical: '#ef4444' };
        let html = `<div class="vlm-row"><span>Density</span><span style="color:${densityColors[a.crowd_density] || '#64748b'}">${a.crowd_density || '—'}</span></div>`;
        html += `<div class="vlm-row"><span>People Est.</span><span>${a.estimated_people ?? '—'}</span></div>`;
        html += `<div class="vlm-row"><span>Stampede Risk</span><span style="color:${riskColors[a.stampede_risk] || '#64748b'}">${a.stampede_risk || '—'}</span></div>`;
        if (a.safety_hazards && a.safety_hazards.length > 0) {
            html += `<div class="vlm-hazards">⚠ ${a.safety_hazards.join(', ')}</div>`;
        }
        if (a.summary) {
            html += `<div class="vlm-summary">${a.summary}</div>`;
        }
        vlmEl.innerHTML = html;
    }

    const modelEl = document.getElementById('vision-model-status');
    const vramStatusEl = document.getElementById('vision-vram-status');

    if (data.model_manager) {
        const mm = data.model_manager;

        if (modelEl && mm.models) {
            let html = '';
            for (const [name, info] of Object.entries(mm.models)) {
                const icon = info.loaded ? '●' : '○';
                const color = info.loaded ? '#22c55e' : '#64748b';
                const title = info.loaded ? `VRAM: ${info.vram_mb}MB | Latency: ${info.last_inference_ms}ms` : 'Not loaded';
                html += `<span title="${title}" style="color:${color};margin-right:8px;font-size:0.7rem;cursor:help;">${icon} ${name}</span>`;
            }
            modelEl.innerHTML = html;
        }

        if (vramStatusEl) {
            const spanEl = vramStatusEl.querySelector('span');
            const barEl = document.getElementById('vram-bar');

            if (mm.gpu_available) {
                spanEl.textContent = `VRAM: ${mm.used_vram_mb}MB / ${mm.total_vram_mb}MB`;
                const pct = Math.min(100, Math.round((mm.used_vram_mb / mm.total_vram_mb) * 100));
                if (barEl) {
                    barEl.style.width = pct + '%';
                    barEl.style.backgroundColor = pct > 90 ? 'var(--los-critical)' : pct > 75 ? 'var(--los-d)' : 'var(--accent-purple)';
                }
            } else {
                spanEl.textContent = 'CPU Mode (No GPU)';
                if (barEl) barEl.style.width = '0%';
            }
        }
    }

    if (data.mode === 'idle') {
        visionStatus.textContent = 'Stopped';
        visionStatus.style.backgroundColor = '';
        visionStatus.style.color = '';
    } else if (data.mode === 'live') {
        visionStatus.textContent = 'Live';
        visionStatus.style.backgroundColor = '#10b981';
        visionStatus.style.color = '#fff';
    } else if (data.mode === 'video' || data.mode === 'batch') {
        visionStatus.textContent = `Scanning ${data.video_name || ''}`;
        visionStatus.style.backgroundColor = '#f59e0b';
        visionStatus.style.color = '#000';
    }
}

// ═══════════════════════════════════════════════════════════════════
// EVENT SETUP — wires to /api/ingress
// ═══════════════════════════════════════════════════════════════════

const PRESETS = {
    weekday:  { archetype: 'gate',      attendance: 18000, start_hour: 8,  scale: 1.0, weekend: false },
    matchday: { archetype: 'concourse', attendance: 45000, start_hour: 16, scale: 1.5, weekend: false },
    sellout:  { archetype: 'concourse', attendance: 60000, start_hour: 17, scale: 2.0, weekend: false },
};

async function applyIngress(partial) {
    try {
        const res = await fetch('/api/ingress', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(partial),
        });
        const data = await res.json();
        refreshIngressReadout();
        return data;
    } catch (err) {
        console.error('[Event Setup] /api/ingress failed:', err);
    }
}

// GET /api/ingress shape: { enabled, config: {sensor_id, source, ...}, current_rate_per_hour, curve }
function updateIngressReadout(data) {
    const el = document.getElementById('es-readout');
    if (!el || !data) return;
    const perHour = data.current_rate_per_hour ?? 0;
    const cfg = data.config || {};
    const src = cfg.sensor_id ?? cfg.source ?? 'sensor data';
    const state = data.enabled === false ? ' · paused' : '';
    el.textContent = `${Math.round(perHour).toLocaleString()} arrivals/hr · ${src}${state}`;
}

function refreshIngressReadout() {
    fetch('/api/ingress').then(r => r.json()).then(updateIngressReadout).catch(() => {});
}

function hourLabel(h) {
    const hh = Math.floor(h), mm = Math.round((h - hh) * 60);
    return `${String(hh).padStart(2, '0')}:${String(mm).padStart(2, '0')}`;
}

function debounce(fn, ms) {
    let t;
    return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
}

function initEventSetup() {
    if (!document.getElementById('es-attendance')) return;

    document.querySelectorAll('.preset-btn').forEach(btn => {
        btn.addEventListener('click', async () => {
            document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            const p = PRESETS[btn.dataset.preset];
            document.getElementById('es-attendance').value = p.attendance;
            document.getElementById('es-hour').value = p.start_hour;
            document.getElementById('es-scale').value = p.scale;
            document.getElementById('es-attendance-val').textContent = p.attendance.toLocaleString();
            document.getElementById('es-hour-val').textContent = hourLabel(p.start_hour);
            document.getElementById('es-scale-val').textContent = p.scale.toFixed(1) + '×';
            await applyIngress(p);
        });
    });

    const debounced = debounce(applyIngress, 250);

    document.getElementById('es-attendance').addEventListener('input', e => {
        document.getElementById('es-attendance-val').textContent = Number(e.target.value).toLocaleString();
        debounced({ attendance: Number(e.target.value) });
    });

    document.getElementById('es-hour').addEventListener('input', e => {
        document.getElementById('es-hour-val').textContent = hourLabel(Number(e.target.value));
        debounced({ start_hour: Number(e.target.value) });
    });

    document.getElementById('es-scale').addEventListener('input', e => {
        document.getElementById('es-scale-val').textContent = Number(e.target.value).toFixed(1) + '×';
        debounced({ scale: Number(e.target.value) });
    });

    document.querySelectorAll('.day-btn').forEach(btn => {
        btn.addEventListener('click', async () => {
            document.querySelectorAll('.day-btn').forEach(b => b.classList.remove('on'));
            btn.classList.add('on');
            await applyIngress({ weekend: btn.dataset.weekend === 'true' });
        });
    });

    let paused = false;
    document.getElementById('es-pause').addEventListener('click', async e => {
        paused = !paused;
        e.target.textContent = paused ? 'Resume arrivals' : 'Pause arrivals';
        await applyIngress({ enabled: !paused });
    });

    refreshIngressReadout();
    setInterval(refreshIngressReadout, 5000);  // rate follows the hourly curve as sim time advances
}

// ═══════════════════════════════════════════════════════════════════
// INIT
// ═══════════════════════════════════════════════════════════════════

initEventSetup();
initWebSocket();
requestAnimationFrame(draw);
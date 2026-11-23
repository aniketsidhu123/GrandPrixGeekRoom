"""
Autopilot — lets the system act on its own when no operator responds, and keeps a
history of everything it did so the operator can review it on return.

How it works
------------
The engine's early-warning list (see PredictionEngine.early_warnings) says which
zones are expected to pile up, and each warning carries a recommended divert
(origin gate -> target gate, pct). Normally an operator presses "Divert".

With Autopilot ON, a recommendation that nobody has acted on for `wait_sec` seconds
of REAL time (the operator's response window, so it does not shrink at 5x sim speed)
is applied automatically. `wait_sec = 0` means "act immediately".

Guard rails (this is a safety system, so autonomy is deliberately narrow):
  * The ONLY thing it ever does is a timed gate divert - the same reversible action
    the operator button performs. It never closes or throttles gates.
  * It only acts on warnings expected within AUTO_ACT_MAX_ETA_SEC AND where the zone
    is already busy right now (AUTO_MIN_LIVE_DENSITY). The forecast tends to be
    pessimistic, so on its own it is only enough to recommend, never to act.
  * It never diverts more than AUTO_MAX_PCT of a gate's arrivals.
  * It never piles diverts up: an origin that is itself receiving a divert is left
    alone, and it will not send arrivals to a gate that is diverting away or is
    already the target of another divert (it picks another open gate, or waits).
  * It starts OFF after every server restart.

Everything it does - and every operator divert - is written to an activity log
(kept in memory, mirrored to a JSONL file so it survives restarts).
"""

import json
import os
import threading
import time
from collections import deque
from typing import Dict, List, Optional

AUTO_ACT_MAX_ETA_SEC = 300.0   # only act on pile-ups expected within 5 minutes
AUTO_MAX_PCT = 70              # never divert more than this share of a gate's arrivals
AUTO_MIN_LIVE_DENSITY = 1.08   # ...and only where it is ALREADY busy now (LoS D, p/m²): the forecast
                               # alone is not enough evidence for the system to act by itself
MAX_WAIT_SEC = 600
LOG_LIMIT = 200


class Autopilot:
    def __init__(self, log_path: Optional[str] = None):
        self._lock = threading.RLock()
        self.enabled = False
        self.wait_sec = 30
        # origin gate id -> {target, pct, since (monotonic), eta_sec, peak, zone, cell}
        self.pending: Dict[str, dict] = {}
        self.log = deque(maxlen=LOG_LIMIT)
        self.seq = 0
        self.ack_seq = 0
        self.log_path = log_path
        self._load()

    # ─────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────

    def _load(self):
        if not self.log_path or not os.path.exists(self.log_path):
            return
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                lines = f.readlines()[-LOG_LIMIT:]
            for line in lines:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self.log.append(entry)
                self.seq = max(self.seq, int(entry.get("id", 0)))
            self.ack_seq = self.seq   # entries from before a restart count as already seen
        except OSError:
            pass

    def _append_to_disk(self, entry: dict):
        if not self.log_path:
            return
        try:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError:
            pass   # history is best-effort; never let logging break the simulation

    # ─────────────────────────────────────────────
    # Log
    # ─────────────────────────────────────────────

    def record(self, kind: str, source: str, message: str, sim_time: float = 0.0,
               details: Optional[dict] = None) -> dict:
        with self._lock:
            self.seq += 1
            entry = {
                "id": self.seq,
                "ts": round(time.time(), 1),
                "sim_time_sec": round(float(sim_time), 1),
                "kind": kind,        # divert | divert_end | autopilot_on | autopilot_off | setting
                "source": source,    # autopilot | operator | system
                "message": message,
                "details": details or {},
            }
            self.log.append(entry)
            self._append_to_disk(entry)
            return entry

    def recent(self, n: int = 100) -> List[dict]:
        with self._lock:
            return list(self.log)[-n:][::-1]   # newest first

    def acknowledge(self):
        with self._lock:
            self.ack_seq = self.seq

    # ─────────────────────────────────────────────
    # Configuration
    # ─────────────────────────────────────────────

    def set_config(self, enabled: Optional[bool] = None, wait_sec: Optional[int] = None,
                   sim_time: float = 0.0):
        with self._lock:
            if wait_sec is not None:
                wait_sec = int(max(0, min(MAX_WAIT_SEC, wait_sec)))
                if wait_sec != self.wait_sec:
                    self.wait_sec = wait_sec
                    if self.enabled:
                        self.record("setting", "operator",
                                    f"Response window set to {wait_sec}s", sim_time)
                    self.pending.clear()

            if enabled is not None and bool(enabled) != self.enabled:
                self.enabled = bool(enabled)
                self.pending.clear()   # a fresh full window after switching on
                if self.enabled:
                    window = "immediately" if self.wait_sec == 0 else f"after {self.wait_sec}s without a response"
                    self.record("autopilot_on", "operator",
                                f"Autopilot turned ON - will divert gates {window}", sim_time)
                else:
                    self.record("autopilot_off", "operator",
                                "Autopilot turned OFF - decisions wait for the operator", sim_time)

    # ─────────────────────────────────────────────
    # Decision loop (called once per simulation tick)
    # ─────────────────────────────────────────────

    def clear_pending(self, origin_gate: str):
        with self._lock:
            self.pending.pop(origin_gate, None)

    def evaluate(self, engine, now: Optional[float] = None):
        now = time.monotonic() if now is None else now
        with self._lock:
            # Most urgent actionable recommendation per origin gate
            wanted: Dict[str, dict] = {}
            for w in engine.forecast_warnings:
                o, t, pct = w.get("origin_gate"), w.get("target_gate"), w.get("divert_pct", 0)
                if not o or not t or not pct or pct <= 0:
                    continue
                if w["eta_sec"] > AUTO_ACT_MAX_ETA_SEC:
                    continue
                if w.get("local_density", 0.0) < AUTO_MIN_LIVE_DENSITY:
                    continue      # forecast only: leave it as a recommendation for the operator
                if o in engine._diverts:      # already being handled (by operator or autopilot)
                    continue
                cur = wanted.get(o)
                if cur is None or (w["eta_sec"], -w["peak_density"]) < (cur["eta_sec"], -cur["peak_density"]):
                    wanted[o] = w

            for o in list(self.pending):
                if o not in wanted:
                    del self.pending[o]       # warning went away, or someone acted

            for o, w in wanted.items():
                p = self.pending.get(o)
                if p is None:
                    p = self.pending[o] = {"since": now}
                p.update({
                    "target": w["target_gate"],
                    "pct": w["divert_pct"],
                    "eta_sec": w["eta_sec"],
                    "peak": w["peak_density"],
                    "cell": (w["cell_x"], w["cell_y"]),
                    "zone": engine.zone_label(w["cell_x"], w["cell_y"]),
                })

            if not self.enabled:
                return

            for o in list(self.pending):
                p = self.pending[o]
                if now - p["since"] < self.wait_sec:
                    continue

                receiving = {d["target"] for d in engine._diverts.values()}
                if o in receiving:                       # this gate is absorbing someone else's arrivals
                    continue
                target = p["target"]
                if target in engine._diverts or target in receiving:
                    # recommended target is taken: use the least-loaded other open gate, or wait
                    options = [
                        g for g in engine.gate_objects
                        if g.gate_id != o and g.gate_id not in engine._diverts and g.gate_id not in receiving
                        and g.action != "CLOSE"
                        and "THROTTL" not in str(g.status) and "RESTRICT" not in str(g.status)
                    ]
                    if not options:
                        continue
                    target = min(options, key=lambda g: (g.queue_length, -g.target_rate_per_sec)).gate_id
                pct = min(int(p["pct"]), AUTO_MAX_PCT)
                ctx = {
                    "zone": p["zone"],
                    "eta_sec": p["eta_sec"],
                    "peak_density": p["peak"],
                    "waited_sec": int(now - p["since"]),
                }
                # divert_gate() also clears the pending entry via clear_pending()
                engine.divert_gate(o, target, pct, source="autopilot", context=ctx)
                self.pending.pop(o, None)

    # ─────────────────────────────────────────────
    # Status (goes out with every state update)
    # ─────────────────────────────────────────────

    def status(self) -> dict:
        now = time.monotonic()
        with self._lock:
            pending = []
            for o, p in self.pending.items():
                left = max(0.0, self.wait_sec - (now - p["since"])) if self.enabled else None
                pending.append({
                    "origin": o,
                    "target": p["target"],
                    "pct": min(int(p["pct"]), AUTO_MAX_PCT),
                    "eta_sec": p["eta_sec"],
                    "zone": p["zone"],
                    "seconds_left": None if left is None else round(left),
                })
            return {
                "enabled": self.enabled,
                "wait_sec": self.wait_sec,
                "max_eta_sec": AUTO_ACT_MAX_ETA_SEC,
                "max_pct": AUTO_MAX_PCT,
                "min_live_density": AUTO_MIN_LIVE_DENSITY,
                "pending": pending,
                "log_seq": self.seq,
                "ack_seq": self.ack_seq,
                "unseen": sum(1 for e in self.log if e["id"] > self.ack_seq),
            }

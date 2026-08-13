"""
Arrival Model — turns real pedestrian sensor curves into gate ingress rates.

The forecaster needs to know how many people are still *coming*, not just how
the people already inside will disperse. This module supplies that term from
measured data rather than a hand-drawn bell curve.

Curves come from the City of Melbourne Pedestrian Counting System (CC BY 4.0),
reduced to three venue zone archetypes by `scripts/build_arrival_profiles.py`.
Each curve is 24 normalised hourly weights summing to 1.0, so it scales to any
attendance figure.

Typical use:

    model = ArrivalModel(archetype="gate", attendance=30_000, start_hour=16.0)
    model.rate_at(0.0)      # people/sec arriving at sim t=0 (16:00)
    model.arrivals_in(0.0, 60.0)   # expected arrivals over the next minute
"""

import json
import os
from typing import List, Optional

_PROFILE_PATH = os.path.join(os.path.dirname(__file__), "data", "arrival_profiles.json")

# Used when the profile file is missing so the simulation still runs. Flat curve.
_FALLBACK_CURVE = [1.0 / 24.0] * 24


class ArrivalModel:
    """
    Maps simulation time onto a real-world hourly arrival curve.

    Args:
        archetype:  "gate", "concourse" or "food" — which sensor profile to use.
        attendance: total people expected across the whole 24h curve.
        start_hour: wall-clock hour the simulation begins at (16.0 = 16:00).
        weekend:    use the weekend curve instead of the weekday one.
        scale:      multiplier for tuning intensity without touching attendance.
    """

    def __init__(
        self,
        archetype: str = "gate",
        attendance: int = 30_000,
        start_hour: float = 16.0,
        weekend: bool = False,
        scale: float = 1.0,
        profile_path: Optional[str] = None,
    ):
        self.archetype = archetype
        self.attendance = attendance
        self.start_hour = start_hour
        self.weekend = weekend
        self.scale = scale

        self.curve, self.meta = self._load_curve(profile_path or _PROFILE_PATH)
        # Fractional-arrival carry, so rates below 1/tick still spawn eventually.
        self._carry = 0.0

    def _load_curve(self, path: str):
        """Load the 24-value curve for this archetype, falling back to flat."""
        try:
            with open(path) as f:
                data = json.load(f)
            entry = data["archetypes"][self.archetype]
            curve = entry["weekend" if self.weekend else "weekday"]
            meta = {
                "source": data.get("source", "unknown"),
                "sensor_id": entry.get("sensor_id"),
                "days_observed": entry.get("days_observed"),
                "peak_hour": entry.get("peak_hour_weekday"),
            }
            return list(curve), meta
        except (OSError, KeyError, json.JSONDecodeError):
            return list(_FALLBACK_CURVE), {"source": "fallback (flat curve)"}

    def hour_at(self, sim_time_sec: float) -> float:
        """Wall-clock hour (0-24, fractional) corresponding to a sim timestamp."""
        return (self.start_hour + sim_time_sec / 3600.0) % 24.0

    def rate_at(self, sim_time_sec: float) -> float:
        """
        Arrivals per second at the given simulation time.

        The hourly curve is linearly interpolated so the rate ramps smoothly
        instead of stepping at each hour boundary.
        """
        hour = self.hour_at(sim_time_sec)
        lo = int(hour) % 24
        hi = (lo + 1) % 24
        frac = hour - int(hour)

        share = self.curve[lo] * (1.0 - frac) + self.curve[hi] * frac
        people_this_hour = share * self.attendance * self.scale
        return people_this_hour / 3600.0

    def arrivals_in(self, sim_time_sec: float, duration_sec: float) -> float:
        """Expected (fractional) arrivals over a window starting at sim_time_sec."""
        return self.rate_at(sim_time_sec) * duration_sec

    def draw_arrivals(self, sim_time_sec: float, duration_sec: float) -> int:
        """
        Whole number of arrivals for one tick, carrying the fractional remainder.

        Without the carry, any rate below one person per tick would truncate to
        zero and the venue would never fill.
        """
        self._carry += self.arrivals_in(sim_time_sec, duration_sec)
        whole = int(self._carry)
        self._carry -= whole
        return whole

    def describe(self) -> dict:
        """Summary for the API/dashboard so the data source is visible in the UI."""
        return {
            "archetype": self.archetype,
            "attendance": self.attendance,
            "start_hour": self.start_hour,
            "weekend": self.weekend,
            "scale": self.scale,
            **self.meta,
        }

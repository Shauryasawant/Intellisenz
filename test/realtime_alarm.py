"""
realtime_alarm.py -- per-reading, tiered tilt alarm for one RFM node.

Call FastAlarm.update() once for every reading the moment it reaches the server. No waiting for a second session.

Levels (fastest first)
    EVENT         TWO consecutive readings are > event_deg from the reference.                   (big move / knock / slip)
  STEP          median of the last k_fast readings (~1 min) is > step_deg from the reference. (clear step)
  SESSION_STEP  same test, but against the first readings of THIS session -> works with NO reference,
                so it also covers the hours after a re-mount, when the long-term reference is not armed yet.
  SMALL_STEP    median of the last k_slow readings (~5 min) stays > small_deg from the reference.
                Single-session: session-to-session wobble is NOT averaged out, so expect more false alarms
                than the 2-session rule. Treat as "check", not "confirmed".

Each alert carries `handling_like` = readings in the retained buffer wander a lot (someone may be moving the unit),
a hint for telling handling from a real event. The field log still has to confirm.

Thresholds must sit well above the node's healthy noise: use the `diagnose()` table of shm_models.py
(win_sigma_*, sess_sigma_*) to set them per node. Defaults suit a node with ~0.02 m/s^2 axis noise.
`SMALL_STEP` is advisory only; its 0.20 degree floor can be below quiet shifts on some epochs.
"""

from __future__ import annotations

from collections import deque

import numpy as np


def _ang(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(a @ b, -1.0, 1.0))))


def _med_unit(vs) -> np.ndarray:
    m = np.median(np.asarray(vs), axis=0)
    return m / np.linalg.norm(m)


class FastAlarm:
    def __init__(self, ref_vec=None, event_deg=2.0, step_deg=0.5, small_deg=0.12,
                 k_fast=6, k_slow=30, base_n=30, small_hold=10, session_gap_s=120.0,
                 handling_spread_deg=1.0):
        self.ref = None if ref_vec is None else np.asarray(ref_vec, float) / np.linalg.norm(ref_vec)
        self.event_deg, self.step_deg, self.small_deg = event_deg, step_deg, small_deg
        self.k_fast, self.k_slow, self.base_n, self.small_hold = k_fast, k_slow, base_n, small_hold
        self.gap, self.handling_spread = session_gap_s, handling_spread_deg
        self.latched = set()
        self.event_run = 0
        self._new_session(None)

    def _new_session(self, t):
        self.buf = deque(maxlen=self.k_slow)
        self.base, self.base_vec = [], None
        self.raised, self.small_run, self.event_run, self.last_t = set(), 0, 0, t

    def rearm(self, ref_vec):
        """Acknowledge current alerts, install a fresh reference, and restart session state."""
        self.ref = np.asarray(ref_vec, float) / np.linalg.norm(ref_vec)
        self.latched = set()
        self.event_run = 0
        self._new_session(None)

    def update(self, t_s: float, acc_xyz):
        """Return a level's alert once until acknowledged with rearm()."""
        acc = np.asarray(acc_xyz, float)
        n = np.linalg.norm(acc)
        if n == 0 or not np.isfinite(n):
            return None
        if self.last_t is None or t_s - self.last_t > self.gap:
            self._new_session(t_s)
        self.last_t = t_s

        u = acc / n
        self.buf.append(u)
        if len(self.base) < self.base_n:
            self.base.append(u)
            if len(self.base) == self.base_n:
                self.base_vec = _med_unit(self.base)

        recent = list(self.buf)[-self.k_fast:]
        med_fast = _med_unit(recent) if len(recent) >= self.k_fast else None
        handling = len(self.buf) >= 3 and max(
            _ang(x, _med_unit(self.buf)) for x in self.buf
        ) > self.handling_spread

        hits = []
        if self.ref is not None:
            a1 = _ang(u, self.ref)
            self.event_run = self.event_run + 1 if a1 > self.event_deg else 0
            if self.event_run >= 2:
                hits.append(("EVENT", a1))
            if med_fast is not None:
                a6 = _ang(med_fast, self.ref)
                if a6 > self.step_deg:
                    hits.append(("STEP", a6))
            if len(self.buf) >= self.k_slow:
                a30 = _ang(_med_unit(self.buf), self.ref)
                self.small_run = self.small_run + 1 if a30 > self.small_deg else 0
                if self.small_run >= self.small_hold:
                    hits.append(("SMALL_STEP", a30))
        if self.base_vec is not None and med_fast is not None and len(self.base) >= self.base_n:
            a_s = _ang(med_fast, self.base_vec)
            if a_s > self.step_deg:
                hits.append(("SESSION_STEP", a_s))

        for level, angle in hits:
            if level not in self.raised and level not in self.latched:
                self.raised.add(level)
                self.latched.add(level)
                return {"t": t_s, "level": level, "angle_deg": round(angle, 3), "handling_like": bool(handling)}
        return None
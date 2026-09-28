"""
fast_refs.py -- connect shm_models_v2 output to realtime_alarm.FastAlarm.

v2 stores a node's armed reference as (ref_u, ref_v): tilt offsets, in degrees, from the segment's gravity
direction g (unit vector). FastAlarm needs the reference as a unit vector in the sensor frame. Rebuilt with
the SAME basis v2 uses (_basis below is copied from shm_models_v2):

    r  ~  g + tan(ref_u) * u + tan(ref_v) * v          (then normalised)

REQUIRED ONE-TIME EDIT in shm_models_v2.py, inside run_all(), in the `row = {...}` dict for epochs,
right after the line   "ref_u": ..., "ref_v": ...,   add:

    "g_x": round(float(ref[0]), 6), "g_y": round(float(ref[1]), 6), "g_z": round(float(ref[2]), 6),

Re-run v2 once; epochs.csv then carries the columns this module reads.

Use
    from fast_refs import load_fast_alarms, replay
    alarms = load_fast_alarms("runs/epochs.csv", "runs/segments.csv")   # {sensor_id: FastAlarm}
    a = alarms["RFM_0002"].update(server_time_seconds, (ax, ay, az))    # on every incoming reading

    # sanity check on history BEFORE going live (counts alerts the healthy data would have caused):
    print(replay(df_one_sensor_clean, alarms["RFM_0002"]))
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from realtime_alarm import FastAlarm


def _basis(ref: np.ndarray):                      # identical to shm_models_v2._basis
    e = np.zeros(3)
    e[np.argmin(np.abs(ref))] = 1.0
    u = np.cross(ref, e)
    u /= np.linalg.norm(u)
    return u, np.cross(ref, u)


def ref_unit_vector(g_xyz, ref_u_deg: float, ref_v_deg: float) -> np.ndarray:
    g = np.asarray(g_xyz, float)
    g = g / np.linalg.norm(g)
    u, v = _basis(g)
    r = g + np.tan(np.radians(ref_u_deg)) * u + np.tan(np.radians(ref_v_deg)) * v
    return r / np.linalg.norm(r)


def thresholds_from_quiet(quiet_max_shift_deg: float | None, floor_deg: float = 0.12) -> dict:
    """Data-driven alarm levels. Single-session small-step alarms must clear the node's own session-to-session
    wobble by a wide margin (synthetic test: 0.12 deg floor vs 0.04 deg/axis wobble => ~10% of healthy sessions
    false-alarmed; 0.2 deg => ~0.2%)."""
    q = float(quiet_max_shift_deg) if quiet_max_shift_deg and np.isfinite(quiet_max_shift_deg) else 0.0
    return {"small_deg": max(0.20, 3.0 * q), "step_deg": max(0.50, 8.0 * q), "event_deg": 2.0}


def load_fast_alarms(epochs_csv: str, segments_csv: str | None = None, **overrides) -> dict[str, FastAlarm]:
    """One FastAlarm per sensor, built from the sensor's LATEST segment and its latest armed epoch.
    A sensor whose latest segment has no armed epoch yet (e.g. just re-mounted) gets ref_vec=None: it is then
    watched only by SESSION_STEP (change inside a session) until v2 arms a reference."""
    ep = pd.read_csv(epochs_csv)
    need = {"g_x", "g_y", "g_z", "ref_u", "ref_v"}
    if not need <= set(ep.columns):
        raise ValueError(f"epochs.csv lacks {sorted(need - set(ep.columns))}: apply the one-time edit described "
                         "at the top of fast_refs.py and re-run shm_models_v2.")
    ep["sensor"] = ep["segment"].str.rsplit("_s", n=1).str[0]
    ep["seg_no"] = ep["segment"].str.rsplit("_s", n=1).str[1].astype(int)

    latest_seg = {}
    if segments_csv:
        sg = pd.read_csv(segments_csv)
        sg["sensor"] = sg["segment"].str.rsplit("_s", n=1).str[0]
        sg["seg_no"] = sg["segment"].str.rsplit("_s", n=1).str[1].astype(int)
        latest_seg = sg.groupby("sensor")["seg_no"].max().to_dict()

    out = {}
    for sensor in sorted(set(ep["sensor"]) | set(latest_seg)):
        e = ep[ep["sensor"] == sensor]
        newest = latest_seg.get(sensor, int(e["seg_no"].max()) if len(e) else None)
        e = e[e["seg_no"] == newest]
        if e.empty:                                   # newest segment exists but never armed
            out[sensor] = FastAlarm(None, **overrides)
            continue
        row = e.sort_values("epoch").iloc[-1]
        kw = thresholds_from_quiet(row.get("quiet_max_shift"))
        kw.update(overrides)
        out[sensor] = FastAlarm(ref_unit_vector((row.g_x, row.g_y, row.g_z), row.ref_u, row.ref_v), **kw)
    return out


def replay(df_sensor: pd.DataFrame, fa: FastAlarm) -> pd.DataFrame:
    """Feed one sensor's cleaned readings (columns server_time, acc_x, acc_y, acc_z; time-sorted) through a FastAlarm
    and return every alert. On healthy history this shows how often the fast levels would have fired."""
    t = (df_sensor["server_time"] - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().to_numpy()
    acc = df_sensor[["acc_x", "acc_y", "acc_z"]].to_numpy(float)
    rows = []
    for ti, a in zip(t, acc):
        r = fa.update(float(ti), a)
        if r:
            r["time"] = pd.Timestamp(ti, unit="s", tz="UTC")
            rows.append(r)
    return pd.DataFrame(rows)
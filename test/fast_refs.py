"""
fast_refs.py -- connect shm_models_v2 output to realtime_alarm.FastAlarm.

v2 stores a node's armed reference as (ref_u, ref_v): tilt offsets, in degrees, from the segment's gravity
direction g (unit vector). FastAlarm needs the reference as a unit vector in the sensor frame. Rebuilt with
the SAME basis v2 uses (_basis below is copied from shm_models_v2):

    r  ~  g + tan(ref_u) * u + tan(ref_v) * v          (then normalised)

Use
    from fast_refs import latest_segment_numbers, load_fast_alarms, replay
    from intellisenz.models.SHM_modelv2 import Cfg, clean_rfm
    d = clean_rfm(rfm)
    latest_seg = latest_segment_numbers(d, Cfg())
    alarms = load_fast_alarms("runs/epochs.csv", latest_seg=latest_seg)
    a = alarms["RFM_0002"].update(server_time_seconds, (ax, ay, az))    # on every incoming reading

    # sanity check on history BEFORE going live (counts alerts the healthy data would have caused):
    print(replay(df_one_sensor_clean, alarms["RFM_0002"]))
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from realtime_alarm import FastAlarm
from intellisenz.models.SHM_modelv2 import Cfg, clean_rfm, split_segments


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
    # SMALL_STEP is advisory only; quiet shifts can exceed the 0.20 degree floor.
    return {"small_deg": max(0.20, 3.0 * q), "step_deg": max(0.50, 8.0 * q), "event_deg": 2.0}


def latest_segment_numbers(d: pd.DataFrame, cfg: Cfg) -> dict[str, int]:
    """Find each sensor's latest orientation segment from cleaned raw readings."""
    out = {}
    for sid, g in d.groupby("sensor_id"):
        gs = split_segments(g.reset_index(drop=True), cfg)
        if len(gs):
            out[sid] = int(gs.segment.max()) + 1
    return out


def load_fast_alarms(epochs_csv: str, latest_seg: dict[str, int] | None = None,
                     **overrides) -> dict[str, FastAlarm]:
    """Build one alarm per sensor using the raw-data latest segment when provided.

    A latest segment without a modeled epoch gets ref_vec=None and is watched only by
    SESSION_STEP until the model has armed a reference for that mounting.
    """
    ep = pd.read_csv(epochs_csv)
    need = {"g_x", "g_y", "g_z", "ref_u", "ref_v"}
    if not need <= set(ep.columns):
        raise ValueError(f"epochs.csv lacks {sorted(need - set(ep.columns))}: apply the one-time edit described "
                         "at the top of fast_refs.py and re-run shm_models_v2.")
    ep["sensor"] = ep["segment"].str.rsplit("_s", n=1).str[0]
    ep["seg_no"] = ep["segment"].str.rsplit("_s", n=1).str[1].astype(int)

    latest_from_epochs = ep.groupby("sensor")["seg_no"].max().to_dict()
    selected_segments = latest_from_epochs if latest_seg is None else latest_seg

    out = {}
    for sensor in sorted(set(ep["sensor"]) | set(selected_segments)):
        e = ep[ep["sensor"] == sensor]
        newest = selected_segments.get(sensor)
        if newest is None:
            out[sensor] = FastAlarm(None, **overrides)
            continue
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
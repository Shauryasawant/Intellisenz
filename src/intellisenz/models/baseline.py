"""
baseline.py -- deliberately simple tilt baseline for RFM nodes.

Purpose: a reference point that every fancier model (SHM_modelv2, LSTM-AE, FastAlarm) must beat.
No ML, no hidden state, every number explainable in one sentence.

Method (per sensor, per orientation segment)
  1. clean + split at re-mounts (reuses SHM_modelv2.clean_rfm / split_segments)
  2. drop the first `settle_h` hours after a mount, drop samples whose |acc| is off
  3. per session: median gravity direction; cut into EPOCHS where consecutive sessions jump > cut_deg
     (the data has level steps of 0.5-2 deg even when 'healthy', so no single reference holds for a segment)
  4. per epoch: reference = pooled median of its first `ref_sessions` sessions;
     shift = angle between a session median and the reference
  5. threshold = max(floor_deg, median + k * 1.4826 * MAD) of the shifts of the TRAIN sessions
     (sessions after the reference sessions, before the chronological split)
  6. alarm = `persist` consecutive test sessions above the threshold

Evaluation (honest, chronological)
  * false alarm rate: share of healthy TEST sessions above the threshold (and share of persist-runs)
  * detection: inject a permanent tilt step of size theta at every test session, record whether and when
    the alarm fires. Injection is a rotation of the session-median vector about an axis perpendicular to
    the reference, which is exact for a rigid re-tilt of the node.
  * comparison rows: fixed 0.5 deg threshold, and the MAD threshold.

Usage
    python -m intellisenz.models.baseline --parquet parsed_output/rfm.parquet --out runs/baseline
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from intellisenz.models.SHM_modelv2 import ACC, Cfg, clean_rfm, split_segments, _seconds, _basis


@dataclass
class BaseCfg:
    session_gap_s: int = 120
    min_session_n: int = 10        # samples for a session to count
    settle_h: float = 6.0          # drop after a mount
    cut_deg: float = 0.3           # a session-to-session jump above this starts a new epoch (level step)
    ref_sessions: int = 2          # first sessions of an epoch that define its reference
    norm_tol: float = 0.30         # per-sample |acc| tolerance vs. segment median
    k_mad: float = 5.0
    floor_deg: float = 0.20        # never alarm below this (0.12 gave 10% false alarms on healthy sessions earlier)
    fixed_deg: float = 0.5         # naive comparison threshold
    persist: int = 2
    train_frac: float = 0.5        # of post-reference sessions
    min_epoch: int = 8             # sessions an epoch needs to be evaluated
    min_train: int = 3
    min_test: int = 3
    steps_deg: tuple = (0.15, 0.2, 0.3, 0.5, 1.0, 2.0)


def _rot(v: np.ndarray, axis: np.ndarray, deg: float) -> np.ndarray:
    """Rodrigues rotation of vectors v (n,3) about a unit axis."""
    a = np.radians(deg)
    return (v * np.cos(a) + np.cross(axis, v) * np.sin(a) + np.outer(v @ axis, axis) * (1 - np.cos(a)))


def _angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.degrees(np.arccos(np.clip(a @ b, -1.0, 1.0)))


def session_medians(gs: pd.DataFrame, cfg: BaseCfg):
    """Return (times, unit medians (n,3), sample counts) for one segment, after settling and norm filtering."""
    t = _seconds(gs["server_time"])
    gs = gs[t >= t[0] + cfg.settle_h * 3600].reset_index(drop=True)
    if len(gs) < cfg.min_session_n:
        return None
    acc = gs[ACC].to_numpy(float)
    nrm = np.linalg.norm(acc, axis=1)
    med = np.median(nrm)
    ok = (nrm > 0) & (np.abs(nrm - med) / med < cfg.norm_tol)
    gs, acc, nrm = gs[ok].reset_index(drop=True), acc[ok], nrm[ok]
    t = _seconds(gs["server_time"])
    unit = acc / nrm[:, None]
    sess = np.concatenate([[0], np.cumsum(np.diff(t) > cfg.session_gap_s)])
    rows = []
    for s in np.unique(sess):
        idx = np.flatnonzero(sess == s)
        if len(idx) < cfg.min_session_n:
            continue
        m = np.median(unit[idx], axis=0)
        rows.append((t[idx[0]], m / np.linalg.norm(m), len(idx), unit[idx]))
    if not rows:
        return None
    return (np.array([r[0] for r in rows]), np.vstack([r[1] for r in rows]),
            np.array([r[2] for r in rows]), [r[3] for r in rows], t[0])


def _first_alarm(hit: np.ndarray, persist: int):
    """Index (into hit) where `persist` consecutive hits complete, else None."""
    run = 0
    for i, h in enumerate(hit):
        run = run + 1 if h else 0
        if run >= persist:
            return i
    return None


def split_epochs(med: np.ndarray, cut_deg: float):
    """Indices of session groups whose consecutive session medians agree within cut_deg."""
    if len(med) < 2:
        return [np.arange(len(med))]
    jump = np.degrees(np.arccos(np.clip((med[1:] * med[:-1]).sum(axis=1), -1, 1)))
    cuts = np.flatnonzero(jump > cut_deg) + 1
    return np.split(np.arange(len(med)), cuts)


def evaluate_epoch(name: str, med, units, idx, cfg: BaseCfg):
    ref = np.median(np.vstack([units[i] for i in idx[:cfg.ref_sessions]]), axis=0)
    ref /= np.linalg.norm(ref)
    post = idx[cfg.ref_sessions:]
    n_train = max(cfg.min_train, int(len(post) * cfg.train_frac))
    if n_train + cfg.min_test > len(post):
        return None, []
    tr, te = post[:n_train], post[n_train:]
    shift = _angle(med, ref)
    s_tr = shift[tr]
    mad = np.median(np.abs(s_tr - np.median(s_tr)))
    thr_mad = max(cfg.floor_deg, float(np.median(s_tr) + cfg.k_mad * 1.4826 * mad))
    axis = _basis(ref)[0]
    out, rows = {}, []
    for label, thr in (("mad", thr_mad), ("fixed", cfg.fixed_deg)):
        h = shift[te] > thr
        out[label] = dict(thr=round(thr, 3), far_single=round(float(h.mean()), 3),
                          false_alarm_in_span=float(_first_alarm(h, cfg.persist) is not None))
        for theta in cfg.steps_deg:
            detected, delays = 0, []
            starts = range(len(te) - cfg.persist + 1)   # a step needs `persist` sessions left to be confirmable
            for start in starts:
                s = shift[te].copy()
                s[start:] = _angle(_rot(med[te][start:], axis, theta), ref)
                a = _first_alarm(s > thr, cfg.persist)
                if a is not None and a >= start:
                    detected += 1
                    delays.append(a - start + 1)
            rows.append(dict(epoch=name, method=label, step_deg=theta, thr=round(thr, 3),
                             detect_rate=round(detected / len(starts), 3),
                             median_delay_sessions=float(np.median(delays)) if delays else np.nan))
    summary = dict(epoch=name, sessions=int(len(idx)), train=int(len(tr)), test=int(len(te)),
                   train_shift_median=round(float(np.median(s_tr)), 3), train_shift_max=round(float(s_tr.max()), 3),
                   test_shift_max=round(float(shift[te].max()), 3),
                   **{f"{k}_{m}": v for m, d in out.items() for k, v in d.items()})
    return summary, rows


def evaluate_segment(name: str, gs: pd.DataFrame, cfg: BaseCfg):
    res = session_medians(gs, cfg)
    if res is None:
        return [], []
    ts, med, cnt, units, t0 = res
    S, R = [], []
    for e, idx in enumerate(split_epochs(med, cfg.cut_deg)):
        if len(idx) < cfg.min_epoch:
            continue
        s, rows = evaluate_epoch(f"{name}_e{e}", med, units, idx, cfg)
        if s is not None:
            S.append(s)
            R.extend(rows)
    return S, R


def run(rfm: pd.DataFrame, cfg: BaseCfg, out: Path | None = None):
    d = clean_rfm(rfm)
    seg_cfg = Cfg(session_gap_s=cfg.session_gap_s, min_session_n=30)
    S, R = [], []
    for sid, g in d.groupby("sensor_id"):
        g = g.sort_values("server_time").reset_index(drop=True)
        seg = split_segments(g, seg_cfg)
        if seg.empty:
            continue
        for k, gs in seg.groupby("segment"):
            s, rows = evaluate_segment(f"{sid}_s{k}", gs, cfg)
            S.extend(s)
            R.extend(rows)
    summ, det = pd.DataFrame(S), pd.DataFrame(R)
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        summ.to_csv(out / "baseline_segments.csv", index=False)
        det.to_csv(out / "baseline_detection.csv", index=False)
        (out / "baseline_cfg.json").write_text(json.dumps(asdict(cfg), indent=2))
    return summ, det


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="parsed_output/rfm.parquet")
    ap.add_argument("--out", default="runs/baseline")
    a = ap.parse_args()
    summ, det = run(pd.read_parquet(a.parquet), BaseCfg(), Path(a.out))
    pd.set_option("display.width", 200)
    print(summ.to_string(index=False))
    if not det.empty:
        agg = det.groupby(["method", "step_deg"]).agg(epochs=("epoch", "nunique"),
                                                        detect_rate=("detect_rate", "mean"),
                                                        median_delay=("median_delay_sessions", "median")).round(3)
        print("\nDetection of an injected permanent step (mean over segments):")
        print(agg.to_string())


if __name__ == "__main__":
    main()
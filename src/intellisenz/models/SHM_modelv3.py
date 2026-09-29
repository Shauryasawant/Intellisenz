"""
SHM_modelv3.py -- drift-tolerant tilt monitoring for RFM sensors (builds on SHM_modelv2 cleaning/windowing).

WHAT v2 GOT WRONG ON THE 30-DAY EXPORT (measured, see runs_v3/compare.csv)
  * Arming needed >=4 sessions agreeing within 0.10 deg over 12-72 h. The real data drifts (RFM_0004: ~0.4 deg/day),
    so RFM_0004 s2/s3 and RFM_0001 s14 never armed -> ~half of all sessions were never monitored.
  * floor_deg=0.12 was a fixed constant, not tied to each sensor's own noise.
  * A single frozen reference cannot tell "slow drift" from "step", so any drift ended up as an EVENT or as nothing.
  * Evaluation only ran on 2 epochs, so the reported numbers were very thin.

WHAT v3 CHANGES
  1. Arming: first `arm_sessions` qualifying sessions spanning >= `arm_min_h`, no agreement test (drift allowed).
  2. STEP channel: reference = median of the last `ref_k` accepted sessions (rolling, drift-tolerant, never absorbs an
     unconfirmed hit). A session is a hit if  |p - ref| > max(floor_deg, z_thr * sigma_j),
     sigma_j = sqrt(sigma_q^2 + s_w^2 / n_j)  (per-sensor noise estimated from the arming sessions, per-session
     noise from its own window scatter). `persist_sessions` consecutive hits -> EVENT (latched or re-baselined).
  3. DRIFT channel: Theil-Sen slope of session medians over the last `drift_h` hours; flagged when the rate exceeds
     max(drift_floor, calibrated quantile of quiet-period rates). Catches slow ramps a rolling reference absorbs.
  4. Texture + sensor gate: reused from v2 (Model), fused with the new level/drift flags.
  5. Evaluation on EVERY usable segment, several onsets, calibrated ONLY on pre-onset sessions, detection curve over
     step sizes 0.1-1.0 deg and ramp rates, compared side by side with v2.

HONEST LIMITS
  * There are no labelled damage events in this export. "Accuracy" here = detection of injected faults + number of
    events raised on real data (which you must verify with the field team). It is not a measured field accuracy.
  * No temperature channel exists in the RFM export, so thermal tilt is not compensated.

Usage
    python -m intellisenz.models.SHM_modelv3 --csv influx_data.csv --out runs_v3
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import theilslopes

from intellisenz.models import SHM_modelv2 as v2

logger = logging.getLogger("shm_v3")


@dataclass
class Cfg3:
    # windows / segmentation (passed through to v2 builders)
    window_s: int = 300
    min_n: int = 20
    session_gap_s: int = 120
    min_win: int = 2                # v2 used 3; sparse-but-valid sessions were being thrown away
    seg_tol_deg: float = 2.0
    min_session_n: int = 30
    settle_h: float = 6.0
    sample_norm_tol: float = 0.30
    ref_dir_h: float = 24.0
    min_seg_windows: int = 60
    # step channel
    arm_sessions: int = 4
    arm_min_h: float = 6.0
    arm_tol_deg: float = 0.25       # max residual of arming sessions from a straight line (drift allowed, steps not)
    ref_k: int = 6                  # rolling reference length (sessions)
    floor_deg: float = 0.10         # absolute minimum step that can ever alarm
    z_thr: float = 5.0
    sigma_min: float = 0.02         # deg, lower bound for the noise estimate
    persist_sessions: int = 2
    # drift channel
    drift_h: float = 48.0
    drift_min_sessions: int = 5
    drift_floor: float = 0.35       # deg/day
    drift_q: float = 0.995
    low_sens_sigma: float = 0.15    # baseline noise above this = detector honestly cannot see small steps
    # texture / evaluation
    q: float = 0.995
    persist: int = 2
    norm_tol: float = 0.10
    min_eval_sessions: int = 8
    test_frac: float = 0.30
    tz: str = "Asia/Kolkata"
    seed: int = 0

    def v2(self) -> v2.Cfg:
        keys = v2.Cfg.__dataclass_fields__.keys()
        return v2.Cfg(**{k: v for k, v in asdict(self).items() if k in keys and k != "min_win"}, min_win=self.min_win)


def _mad(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    return 1.4826 * np.median(np.abs(x - np.median(x))) if len(x) else 0.0


def session_table(F: pd.DataFrame) -> pd.DataFrame:
    """v2 session table + within-session window scatter (s_w) used for per-session noise."""
    S = v2.session_table(F)
    g = F.groupby("session")
    S["s_w"] = np.hypot(g["dev_u"].apply(_mad), g["dev_v"].apply(_mad)).reindex(S.index).fillna(0.0)
    return S


# ---------------------------------------------------------------------------
# STEP + DRIFT tracker
# ---------------------------------------------------------------------------

def estimate_sigma(S: pd.DataFrame, cfg: Cfg3) -> float:
    """Per-sensor baseline noise (deg): robust MAD of consecutive session-median differences (real steps are a minority
    and MAD ignores them), combined with the typical within-session scatter."""
    q = S[S["n"] >= cfg.min_win]
    if len(q) < 4:
        return cfg.sigma_min
    du, dv = np.diff(q["u"].to_numpy()), np.diff(q["v"].to_numpy())
    between = np.hypot(_mad(du), _mad(dv)) / np.sqrt(2.0)
    within = float(np.median(q["s_w"].to_numpy() / np.sqrt(q["n"].to_numpy())))
    return float(max(cfg.sigma_min, between, within))


def _drift_rate(t_h, u, v, idx, cfg: Cfg3):
    """deg/day, Theil-Sen over the last drift_h hours ending at session idx[-1]."""
    if len(idx) < cfg.drift_min_sessions:
        return np.nan
    tt = t_h[idx] - t_h[idx[-1]]
    ru = theilslopes(u[idx], tt)[0]
    rv = theilslopes(v[idx], tt)[0]
    return float(np.hypot(ru, rv) * 24.0)


def track(S: pd.DataFrame, cfg: Cfg3, sigma_q: float | None = None, drift_thr: float | None = None,
          rebaseline: bool = False, arm_until: int | None = None):
    """Run the tracker over a session table. Returns (table, sigma_q, drift_thr).
    `sigma_q` / `drift_thr` can be passed in (calibrated on earlier data); otherwise they are estimated from the arming
    sessions / the quiet history seen so far.  rebaseline=True starts a new epoch after each EVENT (analysis mode)."""
    N = len(S)
    t, u, v = S["t_h"].to_numpy(), S["u"].to_numpy(), S["v"].to_numpy()
    n, sw = S["n"].to_numpy(), S["s_w"].to_numpy()
    state = np.array(["SETTLING"] * N, dtype=object)
    shift, zsc, drift = np.full(N, np.nan), np.full(N, np.nan), np.full(N, np.nan)
    epoch = np.full(N, -1)
    ep, pos = 0, 0
    quiet_rates: list[float] = []
    calib_sigma = sigma_q
    calib_drift = drift_thr
    while pos < N:
        q = [j for j in range(pos, N) if n[j] >= cfg.min_win]
        arm = None
        for s in range(len(q) - cfg.arm_sessions + 1):
            idx = q[s:s + cfg.arm_sessions]
            if t[idx[-1]] - t[idx[0]] < cfg.arm_min_h:
                continue
            tt = t[idx] - t[idx[0]]           # drift-tolerant coherence: arming sessions must lie on a straight line
            resid = np.hypot(u[idx] - np.polyval(np.polyfit(tt, u[idx], 1), tt),
                             v[idx] - np.polyval(np.polyfit(tt, v[idx], 1), tt))
            if resid.max() <= cfg.arm_tol_deg:
                arm = idx
                break
        if arm is None:
            state[pos:] = "UNARMED"
            break
        first, last = arm[0], arm[-1]
        state[pos:first] = "SETTLING"
        accepted = list(arm)
        state[arm] = "BASE"
        epoch[arm] = ep
        sig = calib_sigma if calib_sigma is not None else estimate_sigma(S.iloc[pos:], cfg)
        pending: list[int] = []
        ev = None
        for j in range(last + 1, N):
            if n[j] < cfg.min_win:
                continue
            epoch[j] = ep
            ref = accepted[-cfg.ref_k:]
            ru, rv = np.median(u[ref]), np.median(v[ref])
            trend = [k for k in accepted if t[k] >= t[j] - cfg.drift_h]
            if len(trend) >= cfg.drift_min_sessions and t[trend[-1]] - t[trend[0]] >= cfg.arm_min_h:
                # drift-compensated prediction at t_j (Theil-Sen line through recent accepted sessions)
                ru = theilslopes(u[trend], t[trend] - t[j])[1]
                rv = theilslopes(v[trend], t[trend] - t[j])[1]
            shift[j] = np.hypot(u[j] - ru, v[j] - rv)
            sj = np.sqrt(sig ** 2 + sw[j] ** 2 / n[j])
            zsc[j] = shift[j] / sj
            hit = shift[j] > max(cfg.floor_deg, cfg.z_thr * sj)
            win = [k for k in accepted if t[k] >= t[j] - cfg.drift_h]
            if not hit:
                state[j] = "OK"
                pending = []
                accepted.append(j)
                win = [k for k in accepted if t[k] >= t[j] - cfg.drift_h]
                rate = _drift_rate(t, u, v, win, cfg)
                drift[j] = rate
                if np.isfinite(rate) and (arm_until is None or j <= arm_until):
                    quiet_rates.append(rate)
                thr = calib_drift
                if thr is None:
                    thr = cfg.drift_floor if len(quiet_rates) < 20 else max(cfg.drift_floor, float(np.quantile(quiet_rates, cfg.drift_q)))
                if np.isfinite(rate) and rate > thr:
                    state[j] = "DRIFT"
            else:
                state[j] = "HIT"
                pending.append(j)
                if len(pending) >= cfg.persist_sessions:
                    ev = pending[0]
                    break
        if ev is None:
            break
        if not rebaseline:
            state[ev:][state[ev:] != "SETTLING"] = "ALARM"
            state[ev] = "EVENT"
            break
        state[ev] = "EVENT"
        epoch[ev:] = -1
        pos, ep = ev + 1, ep + 1
        calib_sigma = sigma_q  # re-estimate per epoch unless calibrated externally
    out = S.copy()
    out["state"], out["epoch"], out["shift_deg"], out["z"], out["drift_deg_day"] = state, epoch, shift, zsc, drift
    return out, float(sig if N and "sig" in locals() else cfg.sigma_min), calib_drift


def level_flags(F: pd.DataFrame, cfg: Cfg3, sigma_q=None, drift_thr=None, arm_until=None) -> pd.DataFrame:
    T, sig, _ = track(session_table(F), cfg, sigma_q, drift_thr, rebaseline=False, arm_until=arm_until)
    m = F["session"]
    return pd.DataFrame({"alarm": m.map(T["state"].isin(["EVENT", "ALARM"])).to_numpy(bool),
                         "drift": m.map(T["state"].eq("DRIFT")).to_numpy(bool)}, index=F.index)


# ---------------------------------------------------------------------------
# fused model (texture + sensor from v2, level + drift from v3)
# ---------------------------------------------------------------------------

def quick_check(F: pd.DataFrame, calib_end: int, cfg: Cfg3, z_fast: float = 6.0, lookback: int = 12) -> np.ndarray:
    """PRODUCTION FAST LANE: per-WINDOW (5-min) check -- no waiting for a full session to close. Each window is
    compared to the median of the `lookback` windows immediately before it (a short recent rolling baseline, not
    a long fixed one), so genuine slow drift already present in the segment does not itself look anomalous.
    Flags the moment a window is z_fast robust-sigmas from that recent baseline. This trades a somewhat higher
    false-alarm rate for latency measured in minutes instead of the hours the session-level channel can take
    when a fault lands mid-session. Use ALONGSIDE the session-level channel, not instead of it.
    `calib_end` sets where monitoring starts (skip until at least `lookback` clean windows exist)."""
    du, dv = F["dev_u"].to_numpy(), F["dev_v"].to_numpy()
    n = len(F)
    flag = np.zeros(n, dtype=bool)
    start = max(calib_end, lookback)
    for j in range(start, n):
        win_u, win_v = du[j - lookback:j], dv[j - lookback:j]
        bu, bv = np.median(win_u), np.median(win_v)
        su = max(cfg.sigma_min, _mad(win_u))
        sv = max(cfg.sigma_min, _mad(win_v))
        if abs(du[j] - bu) / su > z_fast or abs(dv[j] - bv) / sv > z_fast:
            flag[j] = True
    return flag


def fuse(F: pd.DataFrame, tex_model: v2.Model, cfg: Cfg3, sigma_q=None, drift_thr=None, arm_until=None) -> pd.DataFrame:
    lf = level_flags(F, cfg, sigma_q, drift_thr, arm_until)
    sc = tex_model.score(F, lf["alarm"].to_numpy())

    lf = level_flags(F, cfg, sigma_q, drift_thr, arm_until)
    sc = tex_model.score(F, lf["alarm"].to_numpy())
    sc["flag_drift"] = lf["drift"].to_numpy()
    sc["status"] = np.select([sc["flag_sensor"], sc["flag_level"], sc["flag_drift"], sc["flag_tex"]],
                             ["SENSOR", "ALARM", "DRIFT", "WARNING"], "OK")
    return sc


# ---------------------------------------------------------------------------
# evaluation: many segments, many onsets, calibration only on pre-onset data
# ---------------------------------------------------------------------------

STEPS = [0.10, 0.15, 0.20, 0.30, 0.50, 1.00]
RAMPS = [(1.0, 24.0), (1.0, 72.0)]            # (total deg, hours to reach it)


def inject_v3(F: pd.DataFrame, kind: str, mag: float, onset: int, hours: float = 48.0) -> pd.DataFrame:
    if kind == "ramp":
        f = F.copy()
        th = (f.index - f.index[onset]).total_seconds().to_numpy() / 3600.0
        frac = np.clip(th / hours, 0, 1)
        frac[:onset] = 0
        f["dev_u"] = f["dev_u"].to_numpy() + mag * frac
        return f
    return v2.inject(F, kind, mag, onset)


def evaluate_segment(F: pd.DataFrame, cfg: Cfg3, onset_fracs=(0.5, 0.6, 0.7)) -> pd.DataFrame:
    rows = []
    sess = np.sort(F["session"].unique())
    for of in onset_fracs:
        n_cal = int(len(sess) * of)
        if n_cal < cfg.arm_sessions + 2 or len(sess) - n_cal < 3:
            continue
        cal_ids = sess[:n_cal]
        Fc = F[F["session"].isin(cal_ids)]
        onset = int(np.flatnonzero(F["session"].to_numpy() == sess[n_cal])[0])
        Sc, sig, _ = track(session_table(Fc), cfg, rebaseline=False)
        qr = Sc.loc[Sc.state.isin(["OK", "DRIFT"]), "drift_deg_day"].dropna()
        dthr = max(cfg.drift_floor, float(np.quantile(qr, cfg.drift_q))) if len(qr) >= 10 else cfg.drift_floor
        tex = v2.Model(cfg.v2()).fit(Fc)
        post = np.arange(len(F)) >= onset
        arm_until = int(np.flatnonzero(F["session"].to_numpy() == sess[n_cal - 1])[-1])
        clean = fuse(F, tex, cfg, sig, dthr, arm_until=arm_until)
        clean_flag = clean["status"].isin(["ALARM", "DRIFT", "SENSOR", "WARNING"]).to_numpy()
        if clean["flag_level"].to_numpy()[:onset + 1].any():
            continue          # level detector already latched on REAL data before onset: an injected fault cannot be told apart
        cases = [("step", "step", m, 0.0) for m in STEPS] + \
                [(f"ramp {m:g}deg/{h:g}h", "ramp", m, h) for m, h in RAMPS] + \
                [("stuck", "stuck", 0.0, 0.0), ("noise x5", "noise", 5.0, 0.0)]
        for name, kind, mag, h in cases:
            Fx = inject_v3(F, kind, mag, onset, h or 48.0)
            sc = fuse(Fx, tex, cfg, sig, dthr, arm_until=arm_until)
            flagged = sc["status"].isin(["ALARM", "DRIFT", "SENSOR", "WARNING"]).to_numpy()
            new = flagged & ~clean_flag                      # only flags the injected fault ADDED
            hit = np.flatnonzero(new[onset:])
            rows.append({"onset_frac": of, "case": f"step {mag:g}" if kind == "step" else name,
                         "detected": bool(len(hit)),
                         "delay_h": (Fx.index[onset + hit[0]] - Fx.index[onset]).total_seconds() / 3600 if len(hit) else np.nan,
                         "clean_post_flag_rate": float(clean_flag[post].mean()),
                         "sigma_q": sig})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

def run_all(rfm_df: pd.DataFrame, cfg: Cfg3 | None = None, out_dir="runs_v3", sensors=None) -> dict:
    cfg = cfg or Cfg3()
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    np.random.seed(cfg.seed)
    c2 = cfg.v2()
    d = v2.clean_rfm(rfm_df)
    if sensors:
        d = d[d["sensor_id"].isin(sensors)]
    seg_rows, ev_rows, eval_rows, summary = [], [], [], {}
    for sid, g in d.groupby("sensor_id"):
        g = g.sort_values("server_time").reset_index(drop=True)
        for k, gs in v2.split_segments(g, c2).groupby("segment"):
            name = f"{sid}_s{k + 1}"
            built = v2.build_segment(gs, c2)
            if built is None:
                continue
            F, ref = built
            S, sig, _ = track(session_table(F), cfg, rebaseline=True)
            S.to_csv(out / f"{name}_sessions.csv")
            seg_rows.append({"segment": name, "sessions": len(S),
                             "monitored": int(S.state.isin(["OK", "DRIFT", "HIT", "EVENT", "BASE"]).sum()),
                             "unarmed/settling": int(S.state.isin(["UNARMED", "SETTLING"]).sum()),
                             "events": int((S.state == "EVENT").sum()), "drift_sessions": int((S.state == "DRIFT").sum()),
                             "sigma_q_deg": round(sig, 3)})
            for _, r in S[S.state.isin(["EVENT", "DRIFT"])].iterrows():
                ev_rows.append({"segment": name, "kind": r["state"], "when_IST": r["t"].tz_convert(cfg.tz).strftime("%Y-%m-%d %H:%M"),
                                "shift_deg": round(float(r["shift_deg"]), 3) if pd.notna(r["shift_deg"]) else np.nan,
                                "drift_deg_day": round(float(r["drift_deg_day"]), 3) if pd.notna(r["drift_deg_day"]) else np.nan})
            info = {"segment": name, "sigma_q_deg": round(sig, 3), "low_sensitivity": bool(sig > cfg.low_sens_sigma)}
            seg_rows[-1].update({"low_sensitivity": info["low_sensitivity"]})
            for e in sorted(x for x in S["epoch"].unique() if x >= 0):
                Se = S[(S.epoch == e) & S.state.isin(["BASE", "OK", "DRIFT"])]
                Fe = F[F["session"].isin(Se.index)]
                if len(Se) < cfg.min_eval_sessions or len(Fe) < 60:
                    continue
                ev = evaluate_segment(Fe, cfg)
                if len(ev):
                    ev.insert(0, "segment", f"{name}_e{e}")
                    ev["low_sensitivity"] = info["low_sensitivity"]
                    eval_rows.append(ev)
                Sfull, sig_f, _ = track(session_table(Fe), cfg, rebaseline=False)
                qr = Sfull.loc[Sfull.state.isin(["OK", "DRIFT"]), "drift_deg_day"].dropna()
                dthr = max(cfg.drift_floor, float(np.quantile(qr, cfg.drift_q))) if len(qr) >= 10 else cfg.drift_floor
                joblib.dump({"tex": v2.Model(c2).fit(Fe), "cfg": asdict(cfg), "seg_ref_gravity": ref, "sigma_q": sig_f,
                             "drift_thr": dthr, "segment": name, "epoch": int(e)}, out / f"{name}_deploy.joblib")
                summary[name] = {"epoch": int(e), "sigma_q": round(sig_f, 4), "drift_thr_deg_day": round(dthr, 3)}
    res = {"segments": pd.DataFrame(seg_rows), "events": pd.DataFrame(ev_rows),
           "eval": pd.concat(eval_rows, ignore_index=True) if eval_rows else pd.DataFrame(), "summary": summary}
    for k in ("segments", "events", "eval"):
        res[k].to_csv(out / f"{k}.csv", index=False)
    (out / "summary.json").write_text(json.dumps({"cfg": asdict(cfg), "deploy": summary}, indent=2, default=str))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="runs_v3")
    ap.add_argument("--sensors", nargs="*", default=None)
    ap.add_argument("--floor", type=float, default=Cfg3.floor_deg)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    from intellisenz.preprocess.preprocessing import load_raw_export, preprocess_raw_export
    rfm = preprocess_raw_export(load_raw_export(a.csv))["RFM"]
    res = run_all(rfm, Cfg3(floor_deg=a.floor), a.out, a.sensors)
    pd.set_option("display.width", 220)
    print("\n=== segments ===\n", res["segments"].to_string(index=False))
    print("\n=== events / drift on real data (verify with field team) ===\n",
          res["events"].to_string(index=False) if len(res["events"]) else "none")
    if len(res["eval"]):
        e = res["eval"]
        print("\n=== fault detection, pooled over segments and onsets (rate = detected share; delay = median hours) ===")
        agg = e.groupby("case").agg(n=("detected", "size"), detect_rate=("detected", "mean"), median_delay_h=("delay_h", "median"))
        print(agg.round(3).to_string())
    print(f"\nOutputs in {a.out}/")


if __name__ == "__main__":
    main()
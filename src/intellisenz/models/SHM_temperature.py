"""
SHM_temperature.py -- physics-based temperature attribution for the SHM pipeline (plugs into SHM_modelv4).

QUESTION IT ANSWERS
    For every EVENT / DRIFT that the tilt (RFM, RFL_01 IMU) and load (RFL kg) trackers raise:
    "Was temperature the reason, partly the reason, or not the reason?"

PHYSICS USED
  1. Temperature acts on the sensor/structure through a THERMAL LAG, not instantly.
     Newton's law of heating:      dTs/dt = (Ta - Ts) / tau
     Ta = air temperature (your file), Ts = temperature the structure/sensor has actually reached,
     tau = thermal time constant (hours). Solved EXACTLY for piecewise-linear Ta (thermal_lag_series).
     tau is not guessed: it is selected from tau_h by BIC (the value the data supports).
  2. Two physical effects are modelled per channel y (tilt u, tilt v, or load kg):
         y = a + drift*t + kT*(Ts - Ts_ref) + kG*(Ta - Ts) [+ one offset per confirmed epoch]
     kT  quasi-static effect of the structure's temperature level: thermal expansion (alpha*dT) and the
         temperature coefficient of offset (TCO) of a MEMS accelerometer / load-cell zero (TCZ).
         For tilt, an accelerometer bias change db (in g) gives a tilt error of about asin(db) rad, so
         typical TCOs of 0.1-1 mg/C mean roughly 0.006-0.06 deg/C -- check YOUR sensor's datasheet.
     kG  thermal GRADIENT effect (sun-heated surface vs cooler core bends the member). The gradient
         driver is G = Ta - Ts = tau*dTs/dt, so it is largest while temperature is changing and it is
         90 degrees out of phase with the level term -- that is why the two are separable.
     drift*t  slow change that temperature does NOT explain (this is what a real ageing/settling trend
         looks like). Epoch offsets stop one real permanent step from distorting kT / kG.
  3. Fit is robust (Huber IRLS). Trust needs ALL of: enough sessions, real temperature swing, temperature
     explains a meaningful share of variance, coefficient is statistically significant (|t| >= min_t) and
     -- for tilt -- physically plausible (|k| below the bound above). Otherwise NOTHING is corrected.
  4. Compensation:   y_comp = y - kT*(Ts - Ts_ref) - kG*(G - G_ref)
     Your own trackers (v3.track for tilt, v4.track_load for load) then run on raw AND y_comp.

VERDICT PER FLAGGED SESSION
    TEMP_EXPLAINED   flagged raw, NOT flagged once temperature is removed
    PARTLY_TEMP      flagged in both, but temperature accounts for >= 50% of the shift
    NOT_TEMP         flagged in both, temperature accounts for < 50%  -> investigate (real movement, handling, ...)
    MASKED_BY_TEMP   flagged only AFTER removing temperature (temperature was hiding a real change)
    Extra columns: temp_share_pct = 1 - (temperature-free shift / raw shift); pred_from_temp = shift the
    fitted physics predicts from the temperature change alone, vs the observed shift.

HONEST LIMITS
  * Correlation is not proof, and the file is (presumably) AIR temperature, not sensor-body temperature.
  * Session-level samples are autocorrelated, so the standard errors / t-values are optimistic.
  * Needs several days with a real day/night swing; short segments come back untrusted (nothing corrected).
  * Sessions outside the temperature file's date range have no temperature and are left uncorrected.
  * The first ~3*tau hours after the file starts are a thermal spin-up (Ts starts equal to Ta).

Standalone use is not needed: SHM_modelv4 calls run_temperature() and print_report() itself.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("shm_temp")
EPOCH0 = pd.Timestamp("1970-01-01", tz="UTC")
HOT = ("EVENT", "DRIFT")
BAD_BIC = 1e18


@dataclass
class TempCfg:
    tz_file: str = "Asia/Kolkata"       # timezone of the temperature file's date/time columns
    tz_out: str = "Asia/Kolkata"        # timezone used when printing event times
    max_gap_h: float = 3.0              # never interpolate across a hole longer than this
    tau_h: tuple = (0.25, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0)   # candidate thermal time constants (hours)
    min_sessions: int = 8               # sessions with temperature needed to fit at all
    min_temp_range_c: float = 3.0       # 5th-95th percentile swing needed to trust a coefficient
    min_r2: float = 0.25                # share of variance temperature must explain
    min_t: float = 3.0                  # |coefficient| / std-error needed
    max_k_tilt_deg_per_c: float = 0.15  # physical plausibility bound for tilt (about 2.6 mg/C)
    huber_c: float = 1.345
    ref_k: int = 6                      # previous sessions used as the reference in attribution
    partly_share: float = 0.5


# ---------------------------------------------------------------------------
# 1. temperature file, alignment, thermal-lag physics
# ---------------------------------------------------------------------------

def _sec(idx) -> np.ndarray:
    idx = pd.DatetimeIndex(idx)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    return np.asarray((idx - EPOCH0).total_seconds(), float)


def load_temperature(path, tz: str = "Asia/Kolkata", date_col="date", time_col="time",
                     temp_col="temperature_C") -> pd.Series:
    """Hourly temperature CSV -> UTC-indexed Series (the sensors' server_time is UTC as well)."""
    d = pd.read_csv(path)
    local = pd.to_datetime(d[date_col].astype(str) + " " + d[time_col].astype(str))
    ts = pd.DatetimeIndex(local).tz_localize(tz, ambiguous="NaT", nonexistent="NaT").tz_convert("UTC")
    s = pd.Series(pd.to_numeric(d[temp_col], errors="coerce").to_numpy(), index=ts)
    s = s[~s.index.isna()].dropna().sort_index()
    return s[~s.index.duplicated()]


def align_temperature(times, temp: pd.Series, max_gap_h: float = 3.0) -> np.ndarray:
    """Linear interpolation of `temp` at `times`; NaN outside the file or across a hole > max_gap_h."""
    t, x, y = _sec(times), _sec(temp.index), temp.to_numpy(float)
    if len(x) < 2:
        return np.full(len(t), np.nan)
    out = np.interp(t, x, y, left=np.nan, right=np.nan)
    i = np.clip(np.searchsorted(x, t), 1, len(x) - 1)
    out[(x[i] - x[i - 1]) > max_gap_h * 3600.0] = np.nan
    return out


def thermal_lag_series(temp: pd.Series, tau_h: float, max_gap_h: float = 3.0) -> pd.Series:
    """First-order thermal response Ts of the structure to air temperature Ta:  dTs/dt = (Ta - Ts)/tau.
    Exact for piecewise-linear Ta: on a step with slope s, the particular solution is Ta - s*tau and the
    homogeneous part decays as exp(-dt/tau). After a data hole longer than max_gap_h, Ts restarts at Ta."""
    x, y = _sec(temp.index) / 3600.0, temp.to_numpy(float)
    ts = np.empty_like(y)
    ts[0] = y[0]
    for i in range(1, len(y)):
        dt = x[i] - x[i - 1]
        if dt <= 0 or dt > max_gap_h:
            ts[i] = y[i]
            continue
        s = (y[i] - y[i - 1]) / dt
        ts[i] = (y[i] - s * tau_h) + (ts[i - 1] - (y[i - 1] - s * tau_h)) * np.exp(-dt / tau_h)
    return pd.Series(ts, index=temp.index)


def session_thermal_features(F: pd.DataFrame, temp: pd.Series, cfg: TempCfg, tau_h: float) -> pd.DataFrame:
    """Per-session mean of Ta (air), Ts (structure, lagged) and G = Ta - Ts (gradient driver).
    F = window frame from v3/v4 (DatetimeIndex + 'session' column)."""
    ta = align_temperature(F.index, temp, cfg.max_gap_h)
    ts = align_temperature(F.index, thermal_lag_series(temp, tau_h, cfg.max_gap_h), cfg.max_gap_h)
    df = pd.DataFrame({"Ta": ta, "Ts": ts, "G": ta - ts})
    return df.groupby(F["session"].to_numpy()).mean()


# ---------------------------------------------------------------------------
# 2. robust physical fit
# ---------------------------------------------------------------------------

def _huber(X: np.ndarray, y: np.ndarray, c: float, iters: int = 40) -> np.ndarray:
    w, beta = np.ones(len(y)), np.zeros(X.shape[1])
    for _ in range(iters):
        sw = np.sqrt(w)
        new = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]
        r = y - X @ new
        s = max(1.4826 * np.median(np.abs(r - np.median(r))), 1e-9)
        a = np.abs(r) / (c * s)
        w = np.where(a <= 1.0, 1.0, 1.0 / a)
        done = np.allclose(new, beta, atol=1e-10)
        beta = new
        if done:
            break
    return beta


def _rob_var(r: np.ndarray) -> float:
    return max(float((1.4826 * np.median(np.abs(r - np.median(r)))) ** 2), 1e-12)


def fit_temp_model(t_h, ts, g, y, cfg: TempCfg, epochs=None, use=None, use_g=True, k_max=None) -> dict:
    """y = a + drift*days + kT*(Ts-Ts_ref) [+ kG*(G-G_ref)] [+ epoch offsets]. Returns a dict."""
    t_h, ts, g, y = (np.asarray(v, float) for v in (t_h, ts, g, y))
    m = np.isfinite(t_h) & np.isfinite(ts) & np.isfinite(y) & (np.isfinite(g) if use_g else True)
    if use is not None:
        m &= np.asarray(use, bool)
    if epochs is not None:
        ep = np.asarray(epochs)
        m &= ep >= 0
    res = {"kT": 0.0, "kG": 0.0, "t_T": np.nan, "t_G": np.nan, "ts_ref": np.nan, "g_ref": 0.0, "r2": 0.0,
           "n": int(m.sum()), "t_range_c": np.nan, "bic": BAD_BIC, "use_g": bool(use_g), "trusted": False,
           "reason": "", "drift_per_day": np.nan, "raw_drift_per_day": np.nan}
    if m.sum() < cfg.min_sessions:
        res["reason"] = f"only {int(m.sum())} sessions with temperature (need {cfg.min_sessions})"
        return res
    n = int(m.sum())
    lo, hi = np.percentile(ts[m], [5, 95])
    res["t_range_c"] = float(hi - lo)
    ts_ref, g_ref = float(np.median(ts[m])), float(np.median(g[m])) if use_g else 0.0
    days = (t_h[m] - t_h[m][0]) / 24.0
    base = [np.ones(n), days]
    if epochs is not None:
        base += [(ep[m] == e).astype(float) for e in sorted(set(ep[m].tolist()))[1:]]
    X0 = np.column_stack(base)
    temp_cols = [ts[m] - ts_ref] + ([g[m] - g_ref] if use_g else [])
    X1 = np.column_stack([X0[:, :2]] + temp_cols + [X0[:, 2:]])
    b0, b1 = _huber(X0, y[m], cfg.huber_c), _huber(X1, y[m], cfg.huber_c)
    v0, v1 = _rob_var(y[m] - X0 @ b0), _rob_var(y[m] - X1 @ b1)
    se = np.sqrt(np.diag(v1 * np.linalg.pinv(X1.T @ X1)))
    res.update(kT=float(b1[2]), kG=float(b1[3]) if use_g else 0.0, ts_ref=ts_ref, g_ref=g_ref,
               r2=float(np.clip(1.0 - v1 / v0, 0.0, 1.0)), t_T=float(b1[2] / max(se[2], 1e-12)),
               t_G=float(b1[3] / max(se[3], 1e-12)) if use_g else np.nan,
               bic=float(n * np.log(v1) + X1.shape[1] * np.log(n)),
               drift_per_day=float(b1[1]), raw_drift_per_day=float(b0[1]))
    tmax = max(abs(res["t_T"]), abs(res["t_G"]) if use_g else 0.0)
    kbig = max(abs(res["kT"]), abs(res["kG"]))
    if res["t_range_c"] < cfg.min_temp_range_c:
        res["reason"] = f"temperature swing {res['t_range_c']:.1f} C < {cfg.min_temp_range_c} C"
    elif res["r2"] < cfg.min_r2:
        res["reason"] = f"temperature explains only {100 * res['r2']:.0f}% of variance"
    elif tmax < cfg.min_t:
        res["reason"] = f"not significant (|t|={tmax:.1f} < {cfg.min_t})"
    elif k_max is not None and kbig > k_max:
        res["reason"] = f"implausibly large sensitivity {kbig:.3f} > physical bound {k_max}"
    else:
        res["trusted"] = True
    return res


def _t_hours(S: pd.DataFrame) -> np.ndarray:
    return ((S["t"] - S["t"].iloc[0]).dt.total_seconds() / 3600.0).to_numpy()


def _fit_all(S, cols, cfg, k_max, epochs=None, use=None, use_g=True) -> dict:
    th, ts, g = _t_hours(S), S["Ts"].to_numpy(float), S["G"].to_numpy(float)
    return {c: fit_temp_model(th, ts, g, S[c].to_numpy(float), cfg, epochs, use, use_g, k_max) for c in cols}


def compensate(S: pd.DataFrame, models: dict, cols) -> pd.DataFrame:
    out = S.copy()
    ts, g = S["Ts"].to_numpy(float), S["G"].to_numpy(float)
    for c in cols:
        out[f"{c}_raw"] = out[c]
        m = models[c]
        if m["trusted"]:
            adj = m["kT"] * (ts - m["ts_ref"]) + (m["kG"] * (g - m["g_ref"]) if m["use_g"] else 0.0)
            out[c] = out[c] - np.where(np.isfinite(adj), adj, 0.0)
    return out


# ---------------------------------------------------------------------------
# 3. raw vs temperature-free tracker run, and the verdict
# ---------------------------------------------------------------------------

def analyze_series(name, F, S, cols, track_fn, temp, cfg: TempCfg, shift_col: str, k_max=None):
    """F: window frame; S: session table; cols: ['u','v'] or ['kg']; track_fn: S -> tracker table with
    'state', 'epoch' and shift_col. Returns tracker tables, models, chosen tau and the verdict table."""
    raw_T = track_fn(S)
    feats = {tau: session_thermal_features(F, temp, cfg, tau) for tau in cfg.tau_h}

    def scan(epochs=None, use=None):
        """pick (tau, with/without gradient term) with the lowest BIC summed over channels"""
        best = None
        for tau, ft in feats.items():
            St = S.assign(Ta=ft["Ta"].reindex(S.index).to_numpy(), Ts=ft["Ts"].reindex(S.index).to_numpy(),
                          G=ft["G"].reindex(S.index).to_numpy())
            for use_g in (False, True):
                mdl = _fit_all(St, cols, cfg, k_max, epochs, use, use_g)
                score = sum(v["bic"] for v in mdl.values())
                if best is None or score < best[0]:
                    best = (score, tau, St, mdl)
        return best[1:]

    tau, St, mdl = scan()                                          # pass 1: all sessions, robust fit
    comp_T = track_fn(compensate(St, mdl, cols))
    # pass 2: drop what temperature could NOT explain and allow one offset per epoch, then refit
    ok = ~comp_T["state"].isin(["HIT", "EVENT", "ALARM", "DRIFT"]).to_numpy()
    tau2, St2, mdl2 = scan(epochs=comp_T["epoch"].to_numpy(), use=ok)
    if all(v["n"] >= cfg.min_sessions for v in mdl2.values()):
        tau, St, mdl = tau2, St2, mdl2
        comp_T = track_fn(compensate(St, mdl, cols))
    S_comp = compensate(St, mdl, cols)

    ts, g = St["Ts"].to_numpy(float), St["G"].to_numpy(float)
    rows = []
    for j in range(len(St)):
        r, c = raw_T["state"].iloc[j], comp_T["state"].iloc[j]
        if r not in HOT and c not in HOT:
            continue
        pt, pg = ts[max(0, j - cfg.ref_k):j], g[max(0, j - cfg.ref_k):j]
        pt, pg = pt[np.isfinite(pt)], pg[np.isfinite(pg)]
        have = np.isfinite(ts[j]) and len(pt) > 0
        dts = float(ts[j] - np.median(pt)) if have else np.nan
        dg = float(g[j] - np.median(pg)) if have and len(pg) else 0.0
        pred = float(np.sqrt(sum(((mdl[c_]["kT"] * dts + mdl[c_]["kG"] * dg) if mdl[c_]["trusted"] else 0.0) ** 2
                                 for c_ in cols))) if have else np.nan
        obs, resid = raw_T[shift_col].iloc[j], comp_T[shift_col].iloc[j]
        share = float(np.clip(1.0 - resid / obs, 0.0, 1.0)) if pd.notna(obs) and pd.notna(resid) and obs > 0 else np.nan
        if r in HOT and c not in HOT:
            verdict = "TEMP_EXPLAINED"
        elif r in HOT and c in HOT:
            verdict = "PARTLY_TEMP" if (np.isfinite(share) and share >= cfg.partly_share) else "NOT_TEMP"
        else:
            verdict = "MASKED_BY_TEMP"
        if not have:
            verdict += " (no temperature data)"
        rows.append({"series": name, "when_IST": St["t"].iloc[j].tz_convert(cfg.tz_out).strftime("%Y-%m-%d %H:%M"),
                     "raw_state": r, "temp_free_state": c, "verdict": verdict,
                     "temp_c": round(float(St["Ta"].iloc[j]), 1) if np.isfinite(St["Ta"].iloc[j]) else np.nan,
                     "d_struct_temp_c": round(dts, 2) if np.isfinite(dts) else np.nan,
                     "observed_shift": round(float(obs), 3) if pd.notna(obs) else np.nan,
                     "temp_free_shift": round(float(resid), 3) if pd.notna(resid) else np.nan,
                     "temp_share_pct": round(100 * share) if np.isfinite(share) else np.nan,
                     "pred_from_temp": round(pred, 3) if np.isfinite(pred) else np.nan})
    return {"raw": raw_T, "comp": comp_T, "S": S_comp, "models": mdl, "tau_h": tau, "attribution": pd.DataFrame(rows)}


def model_rows(name, res, unit) -> list[dict]:
    rows = []
    cov = 100.0 * float(np.isfinite(res["S"]["Ta"].to_numpy(float)).mean()) if len(res["S"]) else 0.0
    for c, m in res["models"].items():
        share = (100 * (1 - abs(m["drift_per_day"]) / abs(m["raw_drift_per_day"]))
                 if m["trusted"] and np.isfinite(m["raw_drift_per_day"]) and abs(m["raw_drift_per_day"]) > 1e-9 else np.nan)
        rows.append({"series": name, "channel": c, "sessions": len(res["S"]), "temp_cov_pct": round(cov),
                     "tau_h": res["tau_h"], "gradient_term": m["use_g"],
                     f"kT_{unit}/C": round(m["kT"], 4), f"kG_{unit}/C": round(m["kG"], 4),
                     "t_kT": round(m["t_T"], 1) if np.isfinite(m["t_T"]) else np.nan, "r2": round(m["r2"], 2),
                     "trusted": m["trusted"],
                     f"drift_raw_{unit}/day": round(m["raw_drift_per_day"], 3) if np.isfinite(m["raw_drift_per_day"]) else np.nan,
                     f"drift_temp_free_{unit}/day": round(m["drift_per_day"], 3) if np.isfinite(m["drift_per_day"]) else np.nan,
                     "drift_temp_share_pct": round(share) if np.isfinite(share) else np.nan,
                     "events_raw": int((res["raw"]["state"] == "EVENT").sum()),
                     "events_temp_free": int((res["comp"]["state"] == "EVENT").sum()),
                     "note": m["reason"]})
    return rows


# ---------------------------------------------------------------------------
# 4. entry point used by SHM_modelv4.main()
# ---------------------------------------------------------------------------

def run_temperature(rfm_df, rfl_df, temp_csv, cfg4, out_dir, tcfg: TempCfg | None = None, v4=None) -> dict:
    """rfm_df: cleaned RFM frame (or None); rfl_df: RFL frame; cfg4: the Cfg4 instance from SHM_modelv4;
    v4: the running SHM_modelv4 module (passed in to avoid a circular import)."""
    from intellisenz.models import SHM_modelv2 as v2, SHM_modelv3 as v3
    if v4 is None:
        from intellisenz.models import SHM_modelv4 as v4
    tcfg = tcfg or TempCfg()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    temp = load_temperature(temp_csv, tcfg.tz_file)
    logger.info("temperature file: %d hourly rows, %s to %s", len(temp), temp.index.min(), temp.index.max())
    c3 = cfg4.v3()
    models, atts, sess_rows = [], [], []

    def add(name, F, S, cols, track_fn, shift_col, unit, k_max):
        res = analyze_series(name, F, S, cols, track_fn, temp, tcfg, shift_col, k_max)
        models.extend(model_rows(name, res, unit))
        if len(res["attribution"]):
            atts.append(res["attribution"])
        Sc = res["S"]
        d = pd.DataFrame({"series": name, "session": Sc.index,
                          "when_IST": [t.tz_convert(tcfg.tz_out).strftime("%Y-%m-%d %H:%M") for t in Sc["t"]],
                          "air_temp_c": Sc["Ta"].round(2).to_numpy(), "struct_temp_c": Sc["Ts"].round(2).to_numpy(),
                          "raw_state": res["raw"]["state"].to_numpy(), "temp_free_state": res["comp"]["state"].to_numpy()})
        for c in cols:
            d[f"{c}_raw"], d[f"{c}_temp_free"] = Sc[f"{c}_raw"].to_numpy(), Sc[c].to_numpy()
        sess_rows.append(d)

    tilt_track = lambda S: v3.track(S, c3, rebaseline=True)[0]
    tilt_kw = dict(cols=["u", "v"], track_fn=tilt_track, shift_col="shift_deg", unit="deg", k_max=tcfg.max_k_tilt_deg_per_c)

    # --- RFM tilt: same construction as SHM_modelv3.run_all ---
    if rfm_df is not None:
        c2 = c3.v2()
        d = v2.clean_rfm(rfm_df)
        for sid, g in d.groupby("sensor_id"):
            g = g.sort_values("server_time").reset_index(drop=True)
            for k, gs in v2.split_segments(g, c2).groupby("segment"):
                built = v2.build_segment(gs, c2)
                if built is None:
                    continue
                F, _ = built
                add(f"{sid}_s{k + 1}", F, v3.session_table(F), **tilt_kw)

    # --- RFL nodes: IMU tilt (like v4.run_tilt_for_node) and load cells (like v4.run_all) ---
    dl = v4.explode_load_cells(v4.clean_rfl(rfl_df))
    for node, g in dl.groupby("sensor_id"):
        if node is None or (isinstance(node, float) and np.isnan(node)):
            continue
        g = g.sort_values("server_time").reset_index(drop=True)
        zero = g["channels_all_zero"].fillna(True).astype(bool) if "channels_all_zero" in g.columns \
            else pd.Series(True, index=g.index)
        if int((~zero).sum()) >= cfg4.min_valid_imu_rows:
            c2 = cfg4.v2()
            gi = g[~zero].reset_index(drop=True)
            for k, gs in v2.split_segments(gi, c2).groupby("segment"):
                built = v2.build_segment(gs, c2)
                if built is None:
                    continue
                F, _ = built
                add(f"{node}_tilt_s{k + 1}", F, v3.session_table(F), **tilt_kw)
        for cell in v4.CELLS:
            Fc = v4.build_load_series(g, cell, cfg4)
            if Fc is None or len(Fc) < cfg4.min_win_load * (cfg4.arm_sessions + 2):
                continue
            add(f"{node}_{cell}", Fc, v4.load_session_table(Fc), ["kg"],
                lambda S: v4.track_load(S, cfg4, rebaseline=True)[0], "shift_kg", "kg", None)

    res = {"models": pd.DataFrame(models),
           "attribution": pd.concat(atts, ignore_index=True) if atts else pd.DataFrame(),
           "sessions": pd.concat(sess_rows, ignore_index=True) if sess_rows else pd.DataFrame(),
           "temperature": {"rows": len(temp), "start_utc": str(temp.index.min()), "end_utc": str(temp.index.max())}}
    for k in ("models", "attribution", "sessions"):
        res[k].to_csv(out / f"temp_{k}.csv", index=False)
    return res


def print_report(res: dict) -> None:
    pd.set_option("display.width", 250, "display.max_columns", 40)
    t = res["temperature"]
    print("\n" + "=" * 78 + "\nTEMPERATURE ATTRIBUTION (physics: thermal lag + level + gradient, robust fit)\n" + "=" * 78)
    print(f"temperature file: {t['rows']} hourly rows, {t['start_utc']} to {t['end_utc']} (UTC)")
    m = res["models"]
    if not len(m):
        print("no series could be analysed")
        return
    keep = [c for c in m.columns if c != "note"]
    print("\n=== temperature model per series/channel (trusted=False -> left uncorrected, see note) ===\n",
          m[keep].to_string(index=False))
    notes = m[(~m["trusted"]) & (m["note"] != "")][["series", "channel", "note"]]
    if len(notes):
        print("\n=== why a model was not trusted ===\n", notes.to_string(index=False))
    a = res["attribution"]
    print("\n=== was each flagged EVENT/DRIFT caused by temperature? ===\n",
          a.to_string(index=False) if len(a) else "no EVENT/DRIFT sessions")
    if len(a):
        v = a["verdict"].str.replace(r" \(no temperature data\)", "", regex=True).value_counts()
        print("\n=== verdict counts ===\n", v.to_string())
    print("\nPer-session temperature features and raw vs temperature-free values: temp_sessions.csv")
"""
SHM_modelv4.py -- adds RFL LOAD-CELL monitoring to v3's tilt monitoring, and fuses the two where
both exist on the same physical node.

WHAT'S NEW VS v3 (v3 only looked at tilt/gravity direction; it never touched the load cells)

  1. LOAD LEVEL CHANNEL  (your requirement 1: "unusual increase/decrease in load")
     A 1D version of v3's step tracker (see track_load()), applied to each load cell's reported
     kg instead of tilt degrees: arms on a straight-line fit of the first few sessions, then flags
     a session whose kg is too far from a rolling, drift-compensated reference.

  2. LOAD DRIFT CHANNEL  (requirement 2: "gradual change over days/weeks")
     Same Theil-Sen slope idea as v3's tilt drift channel, but over kg/day and with a longer
     default lookback (drift_h=7 days, vs v3's 48h) because load trends are naturally slower.

  3. TILT+LOAD FUSION    (requirements 3 and 5: correlate load with tilt, use load as CONTEXT)
     Only possible where load and tilt come from the SAME physical node -- see DATA NOTE below.
     Where both exist, each tilt-flagged window is relabelled:
       tilt anomaly + load ALSO moved around the same time  -> EXPLAINED_BY_LOAD  (expected: something
           was placed/removed, the tilt is a normal mechanical response)
       tilt anomaly + load did NOT move                     -> UNEXPLAINED_TILT   (more concerning: no
           loading event explains the movement -- this is your "normal load + sudden tilt" case)
       load anomaly + tilt did NOT move                     -> LOAD_ONLY          (loading event with
           no structural response, e.g. something heavy but well-supported)

  4. REAL EVENT vs SENSOR GLITCH (requirement 4)
     load_glitch_flags() compares the load cell's own raw ADC noise against its reported kg: kg is
     firmware-computed FROM raw, so a real load change moves both together. A window where raw jumps/
     gets noisy but kg barely moves is flagged as a likely comms/ADC problem, not a real load event.

DATA NOTE -- read this before trusting the fusion output
  Your RFL nodes are D001, D012, RFL_01 (and a stray "RFM_0001"-tagged RFL packet, 77 rows, ignored
  here as noise). D001 and D012 have essentially NO valid IMU (channels_all_zero=True for ~95-100%
  of their rows) -- they give you load-only monitoring (features 1-2), nothing to fuse against.
  RFL_01 is the only node with BOTH a working IMU (~87% valid) and load cells, so it's the only node
  where tilt+load fusion (features 3-5) is actually possible in this export. There is NO metadata
  tying any RFL node to any RFM node (different ID schemes, "site" is "CIDCO" for everyone) -- so
  this file does NOT attempt to correlate RFL_01's load with RFM_0002's tilt, etc. If those sensors
  are in fact on the same physical structure, get that mapping from whoever installed the hardware
  and I can wire it in; guessing it from the data alone would be unfounded.

LOAD-CELL LABEL MAPPING -- INFERRED, not from a datasheet
  Each RFL node reports 2 load cells, but under two different label sets that never overlap in time
  for any node checked: {"5T","7.5T"} used until ~2026-09-05, then {"500","7.5"} afterward. This
  looks like a firmware rename of the same two physical cells, not 4 distinct cells -- confirm with
  the firmware owner. Mapped here as LC_A = {5T, 500}, LC_B = {7.5T, 7.5}.

Usage
    python -m intellisenz.models.SHM_modelv4 --csv influx_data.csv --out runs_v4
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import theilslopes

from intellisenz.models import SHM_modelv2 as v2
from intellisenz.models import SHM_modelv3 as v3

logger = logging.getLogger("shm_v4")

LOAD_LABEL_MAP = {"5T": "LC_A", "500": "LC_A", "7.5T": "LC_B", "7.5": "LC_B"}
CELLS = sorted(set(LOAD_LABEL_MAP.values()))


@dataclass
class Cfg4:
    window_s: int = 300
    session_gap_s: int = 120
    min_n_load: int = 3            # samples per 5-min load window (load can report sparser than IMU)
    min_win_load: int = 2          # windows a session needs to count
    arm_sessions: int = 4
    arm_min_h: float = 6.0
    arm_tol_kg: float | None = None    # None -> self-calibrated as max(1.0, 3*sigma_q)
    ref_k: int = 6
    floor_kg: float = 0.5          # smallest jump that can ever alarm -- tune to your load cells' rating
    z_thr: float = 5.0
    sigma_min_kg: float = 0.05
    persist_sessions: int = 2
    drift_h: float = 7 * 24.0      # load trends are slower than tilt -- default lookback 1 week
    drift_min_sessions: int = 5
    drift_floor_kg_day: float = 1.0
    drift_q: float = 0.995
    min_valid_imu_rows: int = 200  # below this, don't bother running the tilt side for this node
    fuse_tolerance: str = "10min"  # max gap to consider a tilt window and a load session "same time"
    tz: str = "Asia/Kolkata"

    def v2(self) -> v2.Cfg:
        return v2.Cfg(window_s=self.window_s, session_gap_s=self.session_gap_s, tz=self.tz)

    def v3(self) -> "v3.Cfg3":
        return v3.Cfg3(window_s=self.window_s, session_gap_s=self.session_gap_s, tz=self.tz)


# ---------------------------------------------------------------------------
# 1. cleaning + load-cell explode
# ---------------------------------------------------------------------------

def clean_rfl(df: pd.DataFrame) -> pd.DataFrame:
    """Like v2.clean_rfm but WITHOUT the RFM_\\d+ sensor-id filter (RFL ids are D001, RFL_01, ...),
    and without dropping channels_all_zero rows (a dead IMU can still carry a valid load reading)."""
    d = df.copy()
    if "parse_error" in d.columns:
        d = d[~d["parse_error"].fillna(False).astype(bool)]
    d = d.dropna(subset=["server_time"])
    before = len(d)
    d = d.drop_duplicates(subset=["sensor_id", "server_time"])
    logger.info("RFL: dropped %d duplicate rows", before - len(d))
    return d.sort_values(["sensor_id", "server_time"]).reset_index(drop=True)


def explode_load_cells(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten the nested `load_cells` list into flat LC_A_kg/LC_A_raw, LC_B_kg/LC_B_raw columns."""
    kg = {c: np.full(len(df), np.nan) for c in CELLS}
    raw = {c: np.full(len(df), np.nan) for c in CELLS}
    for i, lc in enumerate(df["load_cells"].to_numpy()):
        for cell in (lc or []):
            name = LOAD_LABEL_MAP.get(cell.get("label"))
            if name is None:
                continue
            kg[name][i] = cell.get("weight_kg_reported", np.nan)
            raw[name][i] = cell.get("raw", np.nan)
    out = df.copy()
    for c in CELLS:
        out[f"{c}_kg"], out[f"{c}_raw"] = kg[c], raw[c]
    return out


# ---------------------------------------------------------------------------
# 2. load windows + sessions (1D analogue of v2's tilt windowing)
# ---------------------------------------------------------------------------

def build_load_windows(times, t, kg, raw, cfg: Cfg4):
    sess = np.concatenate([[0], np.cumsum(np.diff(t) > cfg.session_gap_s)])
    t0 = pd.Series(t).groupby(sess).transform("min").to_numpy()
    key = sess * 100000 + ((t - t0) // cfg.window_s).astype(int)
    groups = np.split(np.arange(len(t)), np.flatnonzero(np.diff(key)) + 1)
    rows, starts = [], []
    for idx in groups:
        valid = ~np.isnan(kg[idx])
        if valid.sum() < cfg.min_n_load:
            continue
        ii = idx[valid]
        rows.append((float(np.median(kg[ii])), float(np.nanstd(raw[idx])) if np.isfinite(raw[idx]).any() else 0.0,
                     len(ii), sess[idx[0]]))
        starts.append(ii[0])
    if not rows:
        return None
    return pd.DataFrame(rows, columns=["kg", "raw_std", "n", "session"],
                        index=pd.DatetimeIndex(times.iloc[starts].reset_index(drop=True)))


def build_load_series(g: pd.DataFrame, cell: str, cfg: Cfg4):
    """g: cleaned RFL rows for ONE sensor_id, sorted by time. Returns a windowed load frame or None
    if this cell never reports for this node."""
    kg = g[f"{cell}_kg"].to_numpy(float)
    if np.isnan(kg).all():
        return None
    t = v2._seconds(g["server_time"])
    raw = g[f"{cell}_raw"].to_numpy(float)
    return build_load_windows(g["server_time"], t, kg, raw, cfg)


def load_session_table(F: pd.DataFrame) -> pd.DataFrame:
    F2 = F.assign(t=F.index)
    S = F2.groupby("session").agg(t=("t", "first"), kg=("kg", "median"), n=("kg", "size"))
    S["s_w"] = F.groupby("session")["kg"].apply(lambda x: v3._mad(x.to_numpy())).reindex(S.index).fillna(0.0)
    return S


# ---------------------------------------------------------------------------
# 3. 1D load tracker -- same design as v3.track(), collapsed from (u,v) to a single kg axis
# ---------------------------------------------------------------------------

def track_load(S: pd.DataFrame, cfg: Cfg4, sigma_q=None, drift_thr=None, rebaseline: bool = False):
    N = len(S)
    if N == 0:
        return S.assign(state=[], epoch=[], shift_kg=[], z=[], drift_kg_day=[]), cfg.sigma_min_kg, drift_thr
    t = ((S["t"] - S["t"].iloc[0]).dt.total_seconds() / 3600.0).to_numpy()
    kg, n, sw = S["kg"].to_numpy(), S["n"].to_numpy(), S["s_w"].to_numpy()
    state = np.array(["SETTLING"] * N, dtype=object)
    shift, zsc, drift = np.full(N, np.nan), np.full(N, np.nan), np.full(N, np.nan)
    epoch = np.full(N, -1)
    ep, pos, sig = 0, 0, cfg.sigma_min_kg
    quiet_rates: list[float] = []
    calib_sigma, calib_drift = sigma_q, drift_thr
    while pos < N:
        q = [j for j in range(pos, N) if n[j] >= cfg.min_win_load]
        arm = None
        for s in range(len(q) - cfg.arm_sessions + 1):
            idx = q[s:s + cfg.arm_sessions]
            if t[idx[-1]] - t[idx[0]] < cfg.arm_min_h:
                continue
            tt = t[idx] - t[idx[0]]
            resid = np.abs(kg[idx] - np.polyval(np.polyfit(tt, kg[idx], 1), tt))
            tol = cfg.arm_tol_kg if cfg.arm_tol_kg is not None else max(1.0, 3 * (calib_sigma or cfg.sigma_min_kg))
            if resid.max() <= tol:
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
        sig = calib_sigma if calib_sigma is not None else max(
            cfg.sigma_min_kg, v3._mad(np.diff(kg[arm])) / np.sqrt(2.0),
            float(np.median(sw[arm])) / np.sqrt(max(1, np.median(n[arm]))))
        pending: list[int] = []
        ev = None
        for j in range(last + 1, N):
            if n[j] < cfg.min_win_load:
                continue
            epoch[j] = ep
            ref = accepted[-cfg.ref_k:]
            trend = [k for k in accepted if t[k] >= t[j] - cfg.drift_h]
            if len(trend) >= cfg.drift_min_sessions and t[trend[-1]] - t[trend[0]] >= cfg.arm_min_h:
                rk = theilslopes(kg[trend], t[trend] - t[j])[1]
            else:
                rk = float(np.median(kg[ref]))
            shift[j] = abs(kg[j] - rk)
            sj = np.sqrt(sig ** 2 + sw[j] ** 2 / n[j])
            zsc[j] = shift[j] / sj
            hit = shift[j] > max(cfg.floor_kg, cfg.z_thr * sj)
            if not hit:
                state[j] = "OK"
                pending = []
                accepted.append(j)
                win = [k for k in accepted if t[k] >= t[j] - cfg.drift_h]
                rate = np.nan
                if len(win) >= cfg.drift_min_sessions:
                    rate = float(theilslopes(kg[win], t[win] - t[win[-1]])[0] * 24.0)
                drift[j] = rate
                if np.isfinite(rate):
                    quiet_rates.append(rate)
                thr = calib_drift if calib_drift is not None else (
                    cfg.drift_floor_kg_day if len(quiet_rates) < 20
                    else max(cfg.drift_floor_kg_day, float(np.quantile(np.abs(quiet_rates), cfg.drift_q))))
                if np.isfinite(rate) and abs(rate) > thr:
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
            mask = np.arange(N) >= ev
            keep_settling = state == "SETTLING"
            state[mask & ~keep_settling] = "ALARM"
            state[ev] = "EVENT"
            break
        state[ev] = "EVENT"
        epoch[ev:] = -1
        pos, ep = ev + 1, ep + 1
    out = S.copy()
    out["state"], out["epoch"], out["shift_kg"], out["z"], out["drift_kg_day"] = state, epoch, shift, zsc, drift
    return out, float(sig), calib_drift


def load_glitch_flags(F: pd.DataFrame, z_raw: float = 6.0, lookback: int = 12) -> np.ndarray:
    """requirement 4: raw ADC noise spiking while reported kg barely moves = likely sensor/comms
    glitch, not a real load event (kg is firmware-derived FROM raw, so a real change moves both)."""
    raw = F["raw_std"].to_numpy()
    kg = F["kg"].to_numpy()
    n = len(F)
    flag = np.zeros(n, dtype=bool)
    for j in range(lookback, n):
        base_raw = np.median(raw[j - lookback:j])
        sd_raw = max(1.0, v3._mad(raw[j - lookback:j]))
        kg_delta = abs(kg[j] - np.median(F["kg"].to_numpy()[j - lookback:j]))
        if (raw[j] - base_raw) / sd_raw > z_raw and kg_delta < max(0.2, sd_raw * 0.01):
            flag[j] = True
    return flag


# ---------------------------------------------------------------------------
# 4. tilt (reuse v2/v3 as-is) + tilt<->load fusion
# ---------------------------------------------------------------------------

def run_tilt_for_node(g_imu: pd.DataFrame, cfg: Cfg4, out_dir: Path, node: str):
    """g_imu: IMU-valid rows for one node. Reuses v2 segmentation + v3 tracker unchanged -- the
    same mechanism used for RFM in SHM_modelv3.py, just pointed at an RFL node's own IMU."""
    c2, c3 = cfg.v2(), cfg.v3()
    results = []
    for k, gs in v2.split_segments(g_imu, c2).groupby("segment"):
        built = v2.build_segment(gs, c2)
        if built is None:
            continue
        F, ref = built
        S, sig, _ = v3.track(v3.session_table(F), c3, rebaseline=True)
        qr = S.loc[S.state.isin(["OK", "DRIFT"]), "drift_deg_day"].dropna()
        dthr = max(c3.drift_floor, float(np.quantile(qr, c3.drift_q))) if len(qr) >= 10 else c3.drift_floor
        tex = v2.Model(c2).fit(F)
        sc = v3.fuse(F, tex, c3, sig, dthr)
        sc.index = F.index
        results.append((f"{node}_s{k + 1}", sc, S))
    return results


def fuse_tilt_load(tilt_status: pd.Series, load_state: pd.Series, tolerance: str) -> pd.DataFrame:
    """requirements 3 and 5: nearest-time join of tilt window status and load session state, then
    reclassify tilt anomalies by whether a load change explains them."""
    left = pd.DataFrame({"tilt_status": tilt_status}).sort_index()
    right = pd.DataFrame({"load_state": load_state}).sort_index()
    m = pd.merge_asof(left, right, left_index=True, right_index=True,
                      direction="nearest", tolerance=pd.Timedelta(tolerance))
    tilt_anom = m["tilt_status"].isin(["ALARM", "DRIFT", "SENSOR", "WARNING"])
    load_anom = m["load_state"].isin(["EVENT", "ALARM", "DRIFT"])
    m["fused_status"] = np.select(
        [tilt_anom & load_anom, tilt_anom & ~load_anom, ~tilt_anom & load_anom],
        ["EXPLAINED_BY_LOAD", "UNEXPLAINED_TILT", "LOAD_ONLY"], default="OK")
    return m


# ---------------------------------------------------------------------------
# 5. orchestration
# ---------------------------------------------------------------------------

def run_all(rfl_df: pd.DataFrame, cfg: Cfg4 | None = None, out_dir="runs_v4", rfm_df: pd.DataFrame | None = None) -> dict:
    """rfm_df is optional: pass it to ALSO run v3's full RFM tilt pipeline (segments/events/eval --
    the same tables SHM_modelv3.py prints) and get them back in the returned dict as res["rfm"]."""
    cfg = cfg or Cfg4()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rfm_res = None
    if rfm_df is not None:
        rfm_res = v3.run_all(rfm_df, cfg.v3(), str(out / "rfm"))

    d = explode_load_cells(clean_rfl(rfl_df))

    load_rows, load_events, glitch_rows, fused_rows, node_notes = [], [], [], [], []
    for node, g in d.groupby("sensor_id"):
        if node is None or (isinstance(node, float) and np.isnan(node)):
            continue
        g = g.sort_values("server_time").reset_index(drop=True)
        zero_flag = g["channels_all_zero"].fillna(True).astype(bool) if "channels_all_zero" in g.columns \
            else pd.Series(True, index=g.index)
        n_imu_valid = int((~zero_flag).sum())
        has_imu = n_imu_valid >= cfg.min_valid_imu_rows
        node_notes.append({"node": node, "rows": len(g), "imu_valid_rows": n_imu_valid, "has_tilt": has_imu})

        tilt_results = run_tilt_for_node(g[~zero_flag].reset_index(drop=True), cfg, out, node) if has_imu else []

        for cell in CELLS:
            Fc = build_load_series(g, cell, cfg)
            if Fc is None or len(Fc) < cfg.min_win_load * (cfg.arm_sessions + 2):
                continue
            S = load_session_table(Fc)
            T, sig, dthr = track_load(S, cfg, rebaseline=True)
            T.to_csv(out / f"{node}_{cell}_sessions.csv")
            load_rows.append({"node": node, "cell": cell, "sessions": len(T),
                              "monitored": int(T.state.isin(["OK", "DRIFT", "HIT", "EVENT", "BASE"]).sum()),
                              "unarmed/settling": int(T.state.isin(["UNARMED", "SETTLING"]).sum()),
                              "events": int((T.state == "EVENT").sum()), "drift_sessions": int((T.state == "DRIFT").sum()),
                              "sigma_q_kg": round(sig, 3)})
            for _, r in T[T.state.isin(["EVENT", "DRIFT"])].iterrows():
                load_events.append({"node": node, "cell": cell, "kind": r["state"],
                                    "when_IST": r["t"].tz_convert(cfg.tz).strftime("%Y-%m-%d %H:%M"),
                                    "shift_kg": round(float(r["shift_kg"]), 3) if pd.notna(r["shift_kg"]) else np.nan,
                                    "drift_kg_day": round(float(r["drift_kg_day"]), 3) if pd.notna(r["drift_kg_day"]) else np.nan})

            gf = load_glitch_flags(Fc)
            n_glitch = int(gf.sum())
            if n_glitch:
                glitch_rows.append({"node": node, "cell": cell, "suspected_glitch_windows": n_glitch,
                                    "total_windows": len(Fc), "pct": round(100 * n_glitch / len(Fc), 2)})

            # T is indexed by session id (see load_session_table's groupby("session")); map each
            # load WINDOW to its session's confirmed state, giving a per-window load-state series
            # that can be time-aligned against the tilt side's per-window status.
            state_by_session = T["state"].to_dict()
            load_state_series = pd.Series(Fc["session"].map(state_by_session).to_numpy(), index=Fc.index)

            for res_name, sc, S_tilt in tilt_results:
                fm = fuse_tilt_load(sc["status"], load_state_series, cfg.fuse_tolerance)
                interesting = fm[fm["fused_status"] != "OK"]
                for ts, row in interesting.iterrows():
                    fused_rows.append({"node": node, "segment": res_name, "cell": cell,
                                       "when_IST": ts.tz_convert(cfg.tz).strftime("%Y-%m-%d %H:%M"),
                                       "tilt_status": row["tilt_status"], "load_state": row["load_state"],
                                       "fused_status": row["fused_status"]})

    res = {"nodes": pd.DataFrame(node_notes), "load_segments": pd.DataFrame(load_rows),
           "load_events": pd.DataFrame(load_events), "glitch_summary": pd.DataFrame(glitch_rows),
           "fused_events": pd.DataFrame(fused_rows)}
    for k, df in res.items():
        df.to_csv(out / f"{k}.csv", index=False)
    (out / "summary.json").write_text(json.dumps({"cfg": asdict(cfg)}, indent=2, default=str))
    if rfm_res is not None:
        res["rfm"] = rfm_res
    return res


def _json_safe(obj):
    """Recursively replace NaN/Infinity with None. json.dumps writes bare NaN/Infinity tokens by
    default, which Python's json module can read back but which are NOT valid JSON and every
    browser's JSON.parse correctly rejects -- this is what breaks the dashboard without this step."""
    if isinstance(obj, float):
        return None if (np.isnan(obj) or np.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def export_dashboard_json(res: dict, out_dir: str, path: str = "dashboard_data.json") -> str:
    """Build ONE real-data JSON for a dashboard: every number here is read back from the CSVs this
    pipeline already wrote (segments/events summaries + the actual per-session *_sessions.csv rows).
    Nothing is fabricated, reconstructed, or randomly generated -- if a field would require data this
    pipeline doesn't produce (e.g. temperature), it is simply absent, not invented.
    NOTE: there is no temperature channel anywhere in this dataset (checked directly against the raw
    export, including the otherwise-ignored 'esp32s3' rows) -- any dashboard claiming a temperature
    correction/attribution model is showing fabricated numbers. Don't build that tab from this data."""
    out = Path(out_dir)
    segments = []

    def safe_csv(p: Path) -> pd.DataFrame:
        if not p.exists() or p.stat().st_size == 0:
            return pd.DataFrame()
        try:
            return pd.read_csv(p)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    def load_sessions(csv_path: Path, kind: str):
        S = safe_csv(csv_path)
        if S.empty:
            return []
        cols = [c for c in ["t", "u", "v", "kg", "n", "state", "shift_deg", "shift_kg", "z", "drift_deg_day", "drift_kg_day"]
                if c in S.columns]
        return S[cols].to_dict(orient="records")

    seg_df = safe_csv(out / "rfm" / "segments.csv")
    if not seg_df.empty:
        ev_df = safe_csv(out / "rfm" / "events.csv")
        for _, row in seg_df.iterrows():
            name = row["segment"]
            segments.append({"id": name, "kind": "tilt", "unit": "deg", **row.to_dict(),
                             "events": ev_df[ev_df["segment"] == name].to_dict(orient="records") if not ev_df.empty else [],
                             "session_rows": load_sessions(out / "rfm" / f"{name}_sessions.csv", "tilt")})

    seg_df = safe_csv(out / "load_segments.csv")
    if not seg_df.empty:
        ev_df = safe_csv(out / "load_events.csv")
        for _, row in seg_df.iterrows():
            name = f"{row['node']}_{row['cell']}"
            match = ev_df[(ev_df["node"] == row["node"]) & (ev_df["cell"] == row["cell"])] if not ev_df.empty else pd.DataFrame()
            segments.append({"id": name, "kind": "load", "unit": "kg", **row.to_dict(),
                             "events": match.to_dict(orient="records"),
                             "session_rows": load_sessions(out / f"{row['node']}_{row['cell']}_sessions.csv", "load")})

    payload = {"generated_from": "SHM_modelv4.py run_all() output -- no fabricated or reconstructed values",
              "temperature_data_available": False,
              "fused_events": safe_csv(out / "fused_events.csv").to_dict(orient="records"),
              "glitch_summary": safe_csv(out / "glitch_summary.csv").to_dict(orient="records"),
              "nodes": safe_csv(out / "nodes.csv").to_dict(orient="records"),
              "segments": segments}
    dest = out / path
    dest.write_text(json.dumps(_json_safe(payload), indent=2, default=str))
    return str(dest)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="runs_v4")
    ap.add_argument("--floor_kg", type=float, default=Cfg4.floor_kg)
    ap.add_argument("--skip_rfm", action="store_true", help="skip v3's RFM tilt pipeline, run RFL load-only")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from intellisenz.preprocess.preprocessing import load_raw_export, preprocess_raw_export
    frames = preprocess_raw_export(load_raw_export(a.csv))
    rfm = None if a.skip_rfm else v2.clean_rfm(frames["RFM"])
    cfg = Cfg4(floor_kg=a.floor_kg)
    res = run_all(frames["RFL"], cfg, a.out, rfm_df=rfm)
    pd.set_option("display.width", 220)

    if "rfm" in res:
        print("\n" + "=" * 78 + "\nRFM TILT MONITORING (same engine as SHM_modelv3.py)\n" + "=" * 78)
        print("\n=== segments ===\n", res["rfm"]["segments"].to_string(index=False))
        print("\n=== events / drift on real data (verify with field team) ===\n",
              res["rfm"]["events"].to_string(index=False) if len(res["rfm"]["events"]) else "none")
        if len(res["rfm"]["eval"]):
            e = res["rfm"]["eval"]
            print("\n=== fault detection, pooled over segments and onsets (rate = detected share; delay = median hours) ===")
            agg = e.groupby("case").agg(n=("detected", "size"), detect_rate=("detected", "mean"),
                                        median_delay_h=("delay_h", "median"))
            print(agg.round(3).to_string())

    print("\n" + "=" * 78 + "\nRFL LOAD MONITORING + TILT/LOAD FUSION (new in v4)\n" + "=" * 78)
    print("\n=== nodes ===\n", res["nodes"].to_string(index=False))
    print("\n=== load segments (per node/cell) ===\n",
          res["load_segments"].to_string(index=False) if len(res["load_segments"]) else "none usable")
    print("\n=== load events (verify with field team) ===\n",
          res["load_events"].to_string(index=False) if len(res["load_events"]) else "none")
    print("\n=== suspected load-cell sensor glitches (raw moved, kg didn't) ===\n",
          res["glitch_summary"].to_string(index=False) if len(res["glitch_summary"]) else "none")
    print("\n=== fused tilt+load events (only where both channels exist on the same node) ===\n",
          res["fused_events"].to_string(index=False) if len(res["fused_events"]) else "none")
    dash_path = export_dashboard_json(res, a.out)
    print(f"\nOutputs in {a.out}/  (RFM tilt outputs under {a.out}/rfm/)")
    print(f"Real (non-fabricated) dashboard data written to {dash_path}")


if __name__ == "__main__":
    main()
"""
shm_models_v2.py -- RFM tilt monitoring, redesigned around what the data showed.

WHY A REDESIGN (findings from the runs so far)
  * "Healthy" data contains level steps (0.2-0.7 deg) and re-mounts. A fixed baseline cannot be healthy-stationary.
  * Mahalanobis on tilt LEVEL saturates. Level and texture must be separate channels.
  * A rolling reference forgets a permanent step; a frozen reference with a latched alarm does not.
  * The reference must be built from a time span whose sessions agree, not from "the first 5 sessions".

PIPELINE (per sensor)
  1. clean -> split into ORIENTATION SEGMENTS (session-median gravity direction changes > seg_tol_deg = re-mount)
  2. 5-min windows inside sessions; tilt deviation from the segment's own gravity direction
  3. SESSION-LEVEL DETECTOR (level channel):
        arm: reference = median tilt of >= ref_min_sessions qualifying sessions spanning ref_min_h..ref_max_h,
             accepted only if they agree within ref_agree_deg (else the detector refuses to arm)
        monitor: a qualifying session (>= min_win windows) is 'hit' if |median tilt - reference| > floor_deg;
                 persist_sessions consecutive hits -> EVENT, alarm latched
        healthy-data mode (rebaseline=True): after an EVENT a NEW EPOCH is started (= operator acknowledged),
                 so every quiet stretch becomes its own baseline and you get an event log for the field team
  4. TEXTURE channel (per epoch): robust Mahalanobis + Isolation Forest on level-free features
        [slope_u, slope_v, |acc|, log std]; thresholds from session-grouped out-of-fold scores
  5. sensor gate: stuck (flat std) or |acc| far from healthy value
  6. fusion per window:  SENSOR > ALARM (level latched) > WARNING (texture outlier, 2 windows) > OK
  7. synthetic-fault evaluation on the last 30% of sessions of every epoch that has >= min_epoch_sessions

HONEST LIMITS (printed in the report)
  * Epochs are cut where the level detector fires, so the level channel's own false-alarm rate on clean data is
    zero by construction. What is reported instead is the quiet margin: floor_deg / (largest quiet shift).
  * With a few days per epoch this is a pipeline check, not a performance claim.

Usage
    python -m intellisenz.models.shm_models_v2 --csv influx_data.csv --out runs [--floor 0.12]
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
from sklearn.covariance import LedoitWolf, MinCovDet
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger("shm_v2")
ACC = ["acc_x", "acc_y", "acc_z"]
TEX = ["slope_u", "slope_v", "log_std"]
_EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


@dataclass
class Cfg:
    # windows / sessions
    window_s: int = 300
    min_n: int = 20                 # samples per window
    session_gap_s: int = 120
    min_win: int = 3                # windows a session needs to count for the level channel
    # segmentation
    seg_tol_deg: float = 2.0        # gravity-direction change between sessions that means "re-mounted"
    min_session_n: int = 30         # samples; shorter sessions are handling/transit
    settle_h: float = 6.0           # drop after a segment starts
    sample_norm_tol: float = 0.30
    ref_dir_h: float = 24.0         # hours used for the segment's gravity direction
    min_seg_windows: int = 60
    # level channel
    ref_min_sessions: int = 4
    ref_min_h: float = 12.0
    ref_max_h: float = 72.0
    ref_agree_deg: float = 0.10
    floor_deg: float = 0.12
    persist_sessions: int = 2
    prearm_check: bool = False     # flag sessions BEFORE the reference that differ from it (step during arming)
    # texture channel
    q: float = 0.995
    persist: int = 2
    norm_tol: float = 0.10
    # evaluation
    min_epoch_sessions: int = 10
    test_frac: float = 0.30
    tz: str = "Asia/Kolkata"
    seed: int = 0


# ---------------------------------------------------------------------------
# 1. cleaning + segmentation + windows
# ---------------------------------------------------------------------------

def clean_rfm(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for c in ("parse_error", "channels_all_zero"):
        if c in d.columns:
            d = d[~d[c].fillna(False).astype(bool)]
    d = d[d["sensor_id"].astype(str).str.match(r"^RFM_\d+$")]
    d = d.dropna(subset=["server_time", *ACC])
    before = len(d)
    d = d.drop_duplicates(subset=["sensor_id", "server_time"])
    logger.info("dropped %d duplicate rows", before - len(d))
    return d.sort_values(["sensor_id", "server_time"]).reset_index(drop=True)


def _seconds(ts: pd.Series) -> np.ndarray:
    return (ts - _EPOCH).dt.total_seconds().to_numpy()


def _basis(ref: np.ndarray):
    e = np.zeros(3)
    e[np.argmin(np.abs(ref))] = 1.0
    u = np.cross(ref, e)
    u /= np.linalg.norm(u)
    return u, np.cross(ref, u)


def split_segments(g: pd.DataFrame, cfg: Cfg) -> pd.DataFrame:
    """Label rows with an orientation segment (re-mount = session-median gravity direction jumps)."""
    t = _seconds(g["server_time"])
    acc = g[ACC].to_numpy(float)
    nrm = np.linalg.norm(acc, axis=1)
    sess = np.concatenate([[0], np.cumsum(np.diff(t) > cfg.session_gap_s)])
    u = pd.DataFrame(acc / np.where(nrm > 0, nrm, 1.0)[:, None], columns=["x", "y", "z"])
    u["sess"] = sess
    cnt = u.groupby("sess").size()
    keep = cnt[cnt >= cfg.min_session_n].index
    if len(keep) == 0:
        return g.iloc[0:0].assign(segment=pd.Series(dtype=int))
    m = u.groupby("sess")[["x", "y", "z"]].median().loc[keep].to_numpy()
    m = m / np.linalg.norm(m, axis=1, keepdims=True)
    ang = np.degrees(np.arccos(np.clip((m[1:] * m[:-1]).sum(axis=1), -1, 1)))
    seg = pd.Series(np.concatenate([[0], np.cumsum(ang > cfg.seg_tol_deg)]), index=keep)
    g = g.assign(segment=pd.Series(sess).map(seg).to_numpy())
    return g.dropna(subset=["segment"]).astype({"segment": int})


def build_windows(times, t, du, dv, norm, acc, cfg: Cfg):
    sess = np.concatenate([[0], np.cumsum(np.diff(t) > cfg.session_gap_s)])
    t0 = pd.Series(t).groupby(sess).transform("min").to_numpy()
    key = sess * 100000 + ((t - t0) // cfg.window_s).astype(int)
    groups = np.split(np.arange(len(t)), np.flatnonzero(np.diff(key)) + 1)
    rows, starts = [], []
    for idx in groups:
        if len(idx) < cfg.min_n:
            continue
        tt = (t[idx] - t[idx[0]]) / 3600.0
        if tt[-1] <= 0:
            continue
        rows.append((du[idx].mean(), dv[idx].mean(),
                     np.polyfit(tt, du[idx], 1)[0], np.polyfit(tt, dv[idx], 1)[0],
                     norm[idx].mean(), acc[idx].std(axis=0).max(), len(idx), sess[idx[0]]))
        starts.append(idx[0])
    if not rows:
        return None
    F = pd.DataFrame(rows, columns=["dev_u", "dev_v", "slope_u", "slope_v", "norm", "acc_std_max", "n", "session"],
                     index=pd.DatetimeIndex(times.iloc[starts].reset_index(drop=True)))
    F["log_std"] = np.log(F["acc_std_max"] + 1e-4)
    return F


def build_segment(gs: pd.DataFrame, cfg: Cfg):
    gs = gs.reset_index(drop=True)
    t = _seconds(gs["server_time"])
    gs = gs[t >= t[0] + cfg.settle_h * 3600].reset_index(drop=True)
    if len(gs) < cfg.min_n * cfg.min_seg_windows:
        return None
    acc = gs[ACC].to_numpy(float)
    norm = np.linalg.norm(acc, axis=1)
    med = np.median(norm)
    ok = (norm > 0) & (np.abs(norm - med) / med < cfg.sample_norm_tol)
    gs, acc, norm = gs[ok].reset_index(drop=True), acc[ok], norm[ok]
    t = _seconds(gs["server_time"])
    unit = acc / norm[:, None]
    m = t < t[0] + cfg.ref_dir_h * 3600
    if m.sum() < 50:
        m[:] = True
    ref = np.median(unit[m], axis=0)
    ref /= np.linalg.norm(ref)
    u, v = _basis(ref)
    du = np.degrees(np.arctan2(unit @ u, unit @ ref))
    dv = np.degrees(np.arctan2(unit @ v, unit @ ref))
    F = build_windows(gs["server_time"], t, du, dv, norm, acc, cfg)
    return (F, ref) if F is not None and len(F) >= cfg.min_seg_windows else None


# ---------------------------------------------------------------------------
# 2. session-level (level) detector
# ---------------------------------------------------------------------------

def session_table(F: pd.DataFrame) -> pd.DataFrame:
    F2 = F.assign(t=F.index)
    S = F2.groupby("session").agg(t=("t", "first"), u=("dev_u", "median"), v=("dev_v", "median"),
                                  n=("dev_u", "size")).sort_values("t")
    S["t_h"] = (S["t"] - _EPOCH).dt.total_seconds() / 3600.0
    return S


def _arm(q, t, u, v, cfg: Cfg):
    """Earliest run of qualifying sessions that spans ref_min_h..ref_max_h and agrees within ref_agree_deg."""
    for s in range(len(q)):
        for k in range(cfg.ref_min_sessions, len(q) - s + 1):
            idx = q[s:s + k]
            span = t[idx[-1]] - t[idx[0]]
            if span > cfg.ref_max_h:
                break
            if span < cfg.ref_min_h:
                continue
            ru, rv = np.median(u[idx]), np.median(v[idx])
            if np.max(np.hypot(u[idx] - ru, v[idx] - rv)) <= cfg.ref_agree_deg:
                return ru, rv, idx[-1], idx[0]
    return None


def run_session_detector(S: pd.DataFrame, cfg: Cfg, rebaseline: bool) -> pd.DataFrame:
    """States: SETTLING (not yet part of a reference), BASE (reference sessions), OK, EVENT (level step found),
    ALARM (latched, only when rebaseline=False), UNARMED (no agreeing reference could be built)."""
    N = len(S)
    t, u, v, n = S["t_h"].to_numpy(), S["u"].to_numpy(), S["v"].to_numpy(), S["n"].to_numpy()
    state = np.array(["SETTLING"] * N, dtype=object)
    epoch = np.full(N, -1)
    shift = np.full(N, np.nan)
    refu = np.full(N, np.nan)
    refv = np.full(N, np.nan)
    pos, ep = 0, 0
    while pos < N:
        q = [j for j in range(pos, N) if n[j] >= cfg.min_win]
        arm = _arm(q, t, u, v, cfg)
        if arm is None:
            state[pos:] = "UNARMED"
            epoch[pos:] = -1
            break
        ru, rv, last, first = arm
        state[pos:first] = "SETTLING"
        epoch[pos:first] = -1
        if cfg.prearm_check:
            run = []
            for j in [x for x in range(pos, first) if n[x] >= cfg.min_win] + [None]:
                if j is not None and np.hypot(u[j] - ru, v[j] - rv) > cfg.floor_deg:
                    run.append(j)
                    continue
                if len(run) >= cfg.persist_sessions:
                    state[run[-1]] = "PREARM_SHIFT"
                    shift[run[-1]] = np.hypot(u[run[-1]] - ru, v[run[-1]] - rv)
                run = []
        state[first:last + 1] = "BASE"
        epoch[first:last + 1] = ep
        refu[first:], refv[first:] = ru, rv
        hits, hit_first, ev = 0, None, None
        for j in range(last + 1, N):
            epoch[j], state[j] = ep, "OK"
            if n[j] < cfg.min_win:
                continue
            shift[j] = np.hypot(u[j] - ru, v[j] - rv)
            if shift[j] > cfg.floor_deg:
                hits += 1
                hit_first = j if hits == 1 else hit_first
                if hits >= cfg.persist_sessions:
                    ev = hit_first
                    break
            else:
                hits = 0
        if ev is None:
            if hits > 0:                      # unconfirmed trailing hit: pending, excluded from baseline and margin
                state[hit_first:] = "PENDING"
            break
        if not rebaseline:
            state[ev:] = "ALARM"
            break
        state[ev] = "EVENT"
        epoch[ev:] = -1
        shift[ev + 1:] = np.nan          # stale values from the hit loop must not leak into the next epoch
        refu[ev:], refv[ev:] = np.nan, np.nan
        pos, ep = ev + 1, ep + 1
    out = S.copy()
    out["state"], out["epoch"], out["shift_deg"], out["ref_u"], out["ref_v"] = state, epoch, shift, refu, refv
    return out


# ---------------------------------------------------------------------------
# 3. texture channel
# ---------------------------------------------------------------------------

class _Fold:
    def __init__(self, seed):
        self.seed = seed

    def fit(self, X):
        self.sc = RobustScaler().fit(X)
        self.sc.scale_ = np.maximum(self.sc.scale_, 1e-4)
        Z = self.sc.transform(X)
        try:
            self.cov = MinCovDet(random_state=self.seed).fit(Z)
        except Exception:
            self.cov = LedoitWolf().fit(Z)
        self.iso = IsolationForest(n_estimators=200, random_state=self.seed).fit(Z)
        return self

    def score(self, X):
        Z = self.sc.transform(X)
        return np.sqrt(self.cov.mahalanobis(Z)), -self.iso.score_samples(Z)


class Model:
    """Texture detectors + sensor gate for one epoch. Level flag comes from the session detector."""

    def __init__(self, cfg: Cfg):
        self.cfg = cfg

    def fit(self, F: pd.DataFrame):
        X, g = F[TEX].to_numpy(float), F["session"].to_numpy()
        k = min(5, len(np.unique(g)))
        oof = np.zeros((len(X), 2))
        for tr, va in GroupKFold(k).split(X, groups=g):
            oof[va] = np.column_stack(_Fold(self.cfg.seed).fit(X[tr]).score(X[va]))
        self.thr = np.quantile(oof, self.cfg.q, axis=0)     # session-grouped out-of-fold => honest thresholds
        self.fold = _Fold(self.cfg.seed).fit(X)
        self.norm_med = float(F["norm"].median())
        self.std_med = float(F["acc_std_max"].median())
        return self

    def score(self, F: pd.DataFrame, level: np.ndarray) -> pd.DataFrame:
        m, i = self.fold.score(F[TEX].to_numpy(float))
        raw = (m > self.thr[0]) | (i > self.thr[1])
        tex = pd.Series(raw.astype(int)).rolling(self.cfg.persist, min_periods=1).min().to_numpy().astype(bool)
        flat = F["acc_std_max"].to_numpy() < max(1e-6, 0.02 * self.std_med)
        off = np.abs(F["norm"].to_numpy() - self.norm_med) / self.norm_med > self.cfg.norm_tol
        sensor = flat | off
        status = np.select([sensor, level, tex], ["SENSOR", "ALARM", "WARNING"], "OK")
        return pd.DataFrame({"maha": m, "iso": i, "flag_tex": tex, "flag_level": level, "flag_sensor": sensor,
                             "status": status}, index=F.index)


def level_flags(F: pd.DataFrame, cfg: Cfg) -> np.ndarray:
    S = run_session_detector(session_table(F), cfg, rebaseline=False)
    return F["session"].map(S["state"].eq("ALARM")).to_numpy(bool)


# ---------------------------------------------------------------------------
# 4. synthetic-fault evaluation
# ---------------------------------------------------------------------------

FAULTS = [("step", 0.2), ("step", 0.3), ("step", 1.0), ("ramp", 1.0), ("stuck", 0.0), ("noise", 5.0)]


def inject(F: pd.DataFrame, kind: str, mag: float, onset: int) -> pd.DataFrame:
    f = F.copy()
    n = len(f)

    def col(c):
        return f[c].to_numpy().copy()

    if kind == "step":
        x = col("dev_u"); x[onset:] += mag; f["dev_u"] = x
    elif kind == "ramp":
        x = col("dev_u"); x[onset:] += mag * np.linspace(0, 1, n - onset); f["dev_u"] = x
    elif kind == "stuck":
        for c in ("dev_u", "dev_v", "norm"):
            x = col(c); x[onset:] = x[onset]; f[c] = x
        for c in ("slope_u", "slope_v", "acc_std_max"):
            x = col(c); x[onset:] = 0.0; f[c] = x
    elif kind == "noise":
        for c in ("acc_std_max", "slope_u", "slope_v"):
            x = col(c); x[onset:] *= mag; f[c] = x
    f["log_std"] = np.log(f["acc_std_max"].to_numpy() + 1e-4)
    return f


def evaluate(F: pd.DataFrame, cfg: Cfg):
    sess = np.sort(F["session"].unique())
    n_tv = int(len(sess) * (1 - cfg.test_frac))
    model = Model(cfg).fit(F[F["session"].isin(sess[:n_tv])])
    te = F["session"].isin(sess[n_tv:]).to_numpy()
    first_te = int(np.flatnonzero(te)[0])
    onset = first_te + int(te.sum()) // 3

    def summarize(Fx, label, region):
        sc = model.score(Fx, level_flags(Fx, cfg))
        fused = sc["status"].isin(["WARNING", "ALARM", "SENSOR"]).to_numpy()
        r = {"case": label,
             "level": sc["flag_level"].to_numpy()[region].mean(),
             "texture": sc["flag_tex"].to_numpy()[region].mean(),
             "sensor": sc["flag_sensor"].to_numpy()[region].mean(),
             "alarm+": sc["status"].isin(["ALARM", "SENSOR"]).to_numpy()[region].mean(),
             "any": fused[region].mean()}
        hit = np.flatnonzero(fused[onset:])
        r["delay_h"] = (Fx.index[onset + hit[0]] - Fx.index[onset]).total_seconds() / 3600 if len(hit) else np.nan
        return r

    rows = [summarize(F, "clean: whole test part", te),
            summarize(F, "clean: same period as faults", np.arange(len(F)) >= onset)]
    post = np.arange(len(F)) >= onset
    for kind, mag in FAULTS:
        rows.append(summarize(inject(F, kind, mag, onset), f"{kind} {mag:g}" if kind != "stuck" else "stuck", post))
    return pd.DataFrame(rows).set_index("case"), model


# ---------------------------------------------------------------------------
# 5. orchestration
# ---------------------------------------------------------------------------

def run_all(rfm_df: pd.DataFrame, cfg: Cfg | None = None, out_dir="runs", sensors=None) -> dict:
    cfg = cfg or Cfg()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.random.seed(cfg.seed)
    d = clean_rfm(rfm_df)
    if sensors:
        d = d[d["sensor_id"].isin(sensors)]

    seg_rows, epoch_rows, event_rows, evals, summary = [], [], [], {}, {}
    for sid, g in d.groupby("sensor_id"):
        g = g.sort_values("server_time").reset_index(drop=True)
        gseg = split_segments(g, cfg)
        for k, gs in gseg.groupby("segment"):
            name = f"{sid}_s{k + 1}"
            built = build_segment(gs, cfg)
            if built is None:
                logger.info("%s: too little usable data -> skipped", name)
                continue
            F, ref = built
            S = run_session_detector(session_table(F), cfg, rebaseline=True)
            S.to_csv(out / f"{name}_sessions.csv")
            seg_rows.append({"segment": name, "start": F.index[0].tz_convert(cfg.tz).strftime("%m-%d %H:%M"),
                             "end": F.index[-1].tz_convert(cfg.tz).strftime("%m-%d %H:%M"),
                             "sessions": len(S), "unarmed/settling": int(S.state.isin(["UNARMED", "SETTLING"]).sum()),
                             "events": int((S.state == "EVENT").sum())})
            for _, r in S[S.state.isin(["EVENT", "PREARM_SHIFT"])].iterrows():
                event_rows.append({"segment": name, "kind": r["state"],
                                   "when_IST": r["t"].tz_convert(cfg.tz).strftime("%Y-%m-%d %H:%M"),
                                   "shift_deg": round(float(r["shift_deg"]), 3) if pd.notna(r["shift_deg"]) else np.nan,
                                   "windows": int(r["n"])})
            latest_bundle = None
            for e in sorted(x for x in S["epoch"].unique() if x >= 0):
                Se = S[(S.epoch == e) & S.state.isin(["BASE", "OK"])]
                Fe = F[F["session"].isin(Se.index)]
                qs = Se["shift_deg"].dropna()
                mx = float(qs.max()) if len(qs) else 0.0
                row = {"segment": name, "epoch": int(e),
                       "start": Se["t"].iloc[0].tz_convert(cfg.tz).strftime("%m-%d %H:%M"),
                       "end": Se["t"].iloc[-1].tz_convert(cfg.tz).strftime("%m-%d %H:%M"),
                       "sessions": len(Se), "windows": len(Fe),
                       "ref_u": round(float(Se.ref_u.iloc[0]), 3), "ref_v": round(float(Se.ref_v.iloc[0]), 3),
                       "g_x": round(float(ref[0]), 6), "g_y": round(float(ref[1]), 6), "g_z": round(float(ref[2]), 6),
                       "quiet_max_shift": round(mx, 3),
                       "margin_x": round(cfg.floor_deg / mx, 1) if mx > 0 else np.inf}
                if len(Se) >= cfg.min_epoch_sessions and len(Fe) >= 80:
                    ev, model = evaluate(Fe, cfg)
                    ev.to_csv(out / f"{name}_e{e}_eval.csv")
                    evals[f"{name}_e{e}"] = ev
                    row["evaluated"] = True
                    latest_bundle = {"model": Model(cfg).fit(Fe), "seg_ref_gravity": ref, "level_ref": (row["ref_u"], row["ref_v"]),
                                     "cfg": asdict(cfg), "segment": name, "epoch": int(e)}
                else:
                    row["evaluated"] = False
                epoch_rows.append(row)
            if latest_bundle:
                joblib.dump(latest_bundle, out / f"{name}_deploy.joblib")
                summary[name] = {"deploy_epoch": latest_bundle["epoch"], "level_ref": latest_bundle["level_ref"]}

    res = {"segments": pd.DataFrame(seg_rows), "epochs": pd.DataFrame(epoch_rows),
           "events": pd.DataFrame(event_rows), "eval": evals, "summary": summary}
    for k in ("segments", "epochs", "events"):
        res[k].to_csv(out / f"{k}.csv", index=False)
    (out / "summary.json").write_text(json.dumps({"cfg": asdict(cfg), "deploy": summary}, indent=2, default=str))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sensors", nargs="*", default=None, help="only these sensor ids, e.g. RFM_0002 RFM_0003")

    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--floor", type=float, default=Cfg.floor_deg, help="level-shift alarm threshold, degrees")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    from intellisenz.preprocess.preprocessing import load_raw_export, preprocess_raw_export

    rfm = preprocess_raw_export(load_raw_export(args.csv))["RFM"]
    res = run_all(rfm, Cfg(floor_deg=args.floor, prearm_check=True), args.out, sensors=args.sensors)
    pd.set_option("display.width", 220)
    print("\n=== orientation segments (re-mounts split here) ===")
    print(res["segments"].to_string(index=False) if len(res["segments"]) else "none")
    print("\n=== EVENT LOG: level steps found in 'healthy' data (send this to the field team) ===")
    print(res["events"].to_string(index=False) if len(res["events"]) else "none")
    print("\n=== epochs (each = one stable baseline). margin_x = floor / largest quiet shift; want >= 2 ===")
    print(res["epochs"].to_string(index=False) if len(res["epochs"]) else "none")
    for name, ev in res["eval"].items():
        print(f"\n=== {name}: fraction of windows flagged (clean rows = false alarms; fault rows = detection) ===")
        print(ev.round(3).to_string())
    print(f"\nOutputs written to {args.out}/   (*_deploy.joblib = model of the latest evaluated epoch per segment)")
    print("Reminder: epochs are cut where the level detector fires, so its clean false-alarm rate is 0 by construction;"
          " judge it by margin_x. Few days per epoch => pipeline check, not a performance claim.")


if __name__ == "__main__":
    main()
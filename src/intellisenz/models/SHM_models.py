"""
shm_models.py -- anomaly-detection models for the RFM tilt/gravity stream.

Designed for the data as audited (Sept 2026): ~10 s sampling that arrives in short sessions
(~15-35 min, several per day), 2-5 days of real coverage per node, no fault labels.
Slow signals only (tilt / drift / sensor health). Nothing here does vibration analysis.

Pipeline per node
  1. clean + drop installation-settling period
  2. 5-minute windows inside each session  ->  tilt-deviation features vs the node's own healthy gravity direction
  3. time split: train 70% / val 15% / test 15%  (thresholds come from val, false alarms measured on test)
  4. detectors
        maha   robust Mahalanobis distance      (pattern outliers)
        iso    Isolation Forest                 (pattern outliers, non-Gaussian)   [full tier only]
        shift  session median tilt vs the previous five sessions (slow drift)
        lstm   pooled LSTM autoencoder on the short within-window signature (optional, needs torch)
        sensor rule gating: stuck sensor, |acc| far from the node's healthy value
  5. fusion with persistence -> OK / WATCH / WARNING / ALARM, or SENSOR (sensor fault suppresses structural alerts)
  6. synthetic-fault evaluation on the held-out test part (tilt step, ramp, stuck sensor, noise burst)

Note on the LSTM-AE: its input has the window mean removed, so it learns the *short-term texture* of a node
(noise level, within-window wobble). It is blind to level shifts by design; session-shift detection covers those.

Usage
    python shm_models.py --csv export.csv --out runs/ [--lstm]
or in Python:
    from shm_models import run_all, Cfg
    results = run_all(rfm_df, Cfg(), out_dir="runs")      # rfm_df = preprocess_raw_export(...)["RFM"]
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.covariance import LedoitWolf, MinCovDet
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

try:
    import torch
    import torch.nn as nn

    HAVE_TORCH = True
except Exception: 
    HAVE_TORCH = False

logger = logging.getLogger("shm_models")

ACC = ["acc_x", "acc_y", "acc_z"]
_EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Cfg:
    window_s: int = 300            # feature window length
    min_n: int = 20                # min samples per window (~67% fill at 10 s sampling)
    session_gap_s: int = 120       # gap that starts a new session
    seq_len: int = 24              # LSTM steps per window (window resampled to this length)
    settle_h: float = 24.0         # drop this long after each segment starts (re-mount settling)
    seg_tol_deg: float = 2.0    # session-to-session gravity-direction change that starts a new segment
    min_session_n: int = 60        # sessions with fewer samples (~10 min) are treated as handling/transit and dropped        # drop this many hours after a node's first sample (installation settling)
    sample_norm_tol: float = 0.30  # drop single samples with |acc| this far (fraction) from node median
    train_frac: float = 0.70
    val_frac: float = 0.15
    q: float = 0.995               # threshold quantile on the healthy validation scores
    persist: int = 2               # consecutive windows a vote level must hold
    session_shift_floor_deg: float = 0.11
    norm_tol: float = 0.10         # sensor gate: window |acc| vs healthy value
    full_min_train: int = 200      # train windows needed for full tier (maha + iso [+ lstm])
    light_min_train: int = 40      # train windows needed for light tier (maha + session shift)
    tz: str = "Asia/Kolkata"
    seed: int = 0


# ---------------------------------------------------------------------------
# 1. Cleaning, geometry, windowing
# ---------------------------------------------------------------------------

def clean_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows that carry a real, attributable RFM measurement."""
    d = df.copy()
    for c in ("parse_error", "channels_all_zero"):
        if c in d.columns:
            d = d[~d[c].fillna(False).astype(bool)]
    d = d[d["sensor_id"].astype(str).str.match(r"^RFM_\d+$")]
    d = d.dropna(subset=["server_time", *ACC])
    return d.sort_values(["sensor_id", "server_time"]).reset_index(drop=True)


def _seconds(ts: pd.Series) -> np.ndarray:
    return (ts - _EPOCH).dt.total_seconds().to_numpy()


def _basis(ref: np.ndarray):
    """Two unit vectors u, v perpendicular to ref (no singularity for any mounting orientation)."""
    e = np.zeros(3)
    e[np.argmin(np.abs(ref))] = 1.0
    u = np.cross(ref, e)
    u /= np.linalg.norm(u)
    v = np.cross(ref, u)
    return u, v


@dataclass
class NodeData:
    name: str
    feat: pd.DataFrame        # one row per window, indexed by window start (UTC)
    seq: np.ndarray           # (n_windows, seq_len, 3): mean-removed [dev_u, dev_v, |acc|] inside each window
    ref: np.ndarray           # healthy gravity direction (unit vector, sensor frame)
    t_tr_end: pd.Timestamp
    t_va_end: pd.Timestamp
    tier: str

    def masks(self):
        i = self.feat.index
        tr = i < self.t_tr_end
        va = (i >= self.t_tr_end) & (i < self.t_va_end)
        te = i >= self.t_va_end
        return tr, va, te


def _build_windows(times: pd.Series, t, dev_u, dev_v, norm, acc, cfg: Cfg):
    sess = np.concatenate([[0], np.cumsum(np.diff(t) > cfg.session_gap_s)])
    t0 = pd.Series(t).groupby(sess).transform("min").to_numpy()
    key = sess * 100000 + ((t - t0) // cfg.window_s).astype(int)
    groups = np.split(np.arange(len(t)), np.flatnonzero(np.diff(key)) + 1)

    rows, seqs, starts = [], [], []
    xi = np.linspace(0, 1, cfg.seq_len)
    for idx in groups:
        if len(idx) < cfg.min_n:
            continue
        tt = (t[idx] - t[idx[0]]) / 3600.0
        if tt[-1] <= 0:
            continue
        u, v, nm = dev_u[idx], dev_v[idx], norm[idx]
        rows.append((u.mean(), v.mean(), np.polyfit(tt, u, 1)[0], np.polyfit(tt, v, 1)[0],
                     nm.mean(), acc[idx].std(axis=0).max(), len(idx), sess[idx[0]]))
        k = np.linspace(0, 1, len(idx))
        seqs.append(np.stack([np.interp(xi, k, u - u.mean()),
                              np.interp(xi, k, v - v.mean()),
                              np.interp(xi, k, nm - nm.mean())], axis=1))
        starts.append(idx[0])
    if not rows:
        return None, None
    feat = pd.DataFrame(rows, columns=["dev_u", "dev_v", "slope_u", "slope_v", "norm",
                                       "acc_std_max", "n", "session"],
                        index=pd.DatetimeIndex(times.iloc[starts].to_numpy(), tz="UTC"))
    return feat, np.asarray(seqs, dtype="float32")


def split_segments(g: pd.DataFrame, cfg: Cfg) -> pd.DataFrame:
    """Label each row with an orientation segment. Sessions whose median gravity direction differs from the
    previous kept session by more than seg_tol_deg start a new segment. Short sessions are dropped."""
    t = _seconds(g["server_time"])
    acc = g[ACC].to_numpy(float)
    nrm = np.linalg.norm(acc, axis=1)
    sess = np.concatenate([[0], np.cumsum(np.diff(t) > cfg.session_gap_s)])
    u = pd.DataFrame(acc / np.where(nrm > 0, nrm, 1.0)[:, None], columns=["x", "y", "z"])
    u["sess"] = sess
    cnt = u.groupby("sess").size()
    med = u.groupby("sess")[["x", "y", "z"]].median().loc[cnt[cnt >= cfg.min_session_n].index]
    if med.empty:
        return g.iloc[0:0].assign(segment=pd.Series(dtype=int))
    m = med.to_numpy()
    m = m / np.linalg.norm(m, axis=1, keepdims=True)
    ang = np.degrees(np.arccos(np.clip((m[1:] * m[:-1]).sum(axis=1), -1, 1)))
    seg = pd.Series(np.concatenate([[0], np.cumsum(ang > cfg.seg_tol_deg)]), index=med.index)
    g = g.assign(segment=pd.Series(sess).map(seg).to_numpy())
    return g.dropna(subset=["segment"]).astype({"segment": int})


def prepare_node(name: str, g: pd.DataFrame, cfg: Cfg) -> list[NodeData]:
    g = g.sort_values("server_time").reset_index(drop=True)
    g = split_segments(g, cfg)
    out = []
    for k, gs in g.groupby("segment"):
        seg_name = f"{name}_s{k + 1}"
        gs = gs.reset_index(drop=True)
        t = _seconds(gs["server_time"])
        gs = gs[t >= t[0] + cfg.settle_h * 3600].reset_index(drop=True)
        nd = _prepare_segment(seg_name, gs, cfg)
        if nd is not None:
            out.append(nd)
    return out


def _prepare_segment(name: str, g: pd.DataFrame, cfg: Cfg) -> NodeData | None:
    if len(g) < cfg.min_n * cfg.light_min_train:
        logger.info("%s: too little data (%d rows) -> skipped", name, len(g))
        return None

    acc = g[ACC].to_numpy(float)
    norm = np.linalg.norm(acc, axis=1)
    med = np.median(norm)
    ok = (norm > 0) & (np.abs(norm - med) / med < cfg.sample_norm_tol)
    g, acc, norm = g[ok].reset_index(drop=True), acc[ok], norm[ok]
    t = _seconds(g["server_time"])

    sess = np.concatenate([[0], np.cumsum(np.diff(t) > cfg.session_gap_s)])
    ns = sess[-1] + 1
    if ns < 6:
        logger.info("%s: only %d sessions -> skipped", name, ns)
        return None
    i1 = int(np.searchsorted(sess, int(ns * cfg.train_frac)))
    i2 = int(np.searchsorted(sess, int(ns * (cfg.train_frac + cfg.val_frac))))
    unit = acc / norm[:, None]
    ref = np.median(unit[:i1], axis=0)
    ref /= np.linalg.norm(ref)
    u, v = _basis(ref)
    dev_u = np.degrees(np.arctan2(unit @ u, unit @ ref))
    dev_v = np.degrees(np.arctan2(unit @ v, unit @ ref))

    feat, seq = _build_windows(g["server_time"], t, dev_u, dev_v, norm, acc, cfg)
    if feat is None:
        return None
    t_tr_end = pd.to_datetime(t[i1], unit="s", utc=True)
    t_va_end = pd.to_datetime(t[i2], unit="s", utc=True)
    nd = NodeData(name, feat, seq, ref, t_tr_end, t_va_end, tier="")
    tr, va, te = nd.masks()
    if min(va.sum(), te.sum()) < 10 or tr.sum() < cfg.light_min_train:
        logger.info("%s: not enough windows (train %d, val %d, test %d) -> skipped",
                    name, tr.sum(), va.sum(), te.sum())
        return None
    nd.tier = "full" if tr.sum() >= cfg.full_min_train else "light"
    return nd


def segment_table(nodes: dict) -> pd.DataFrame:
    rows = []
    for name, nd in nodes.items():
        i = nd.feat.index
        rows.append({"segment": name, "start": i[0].strftime("%m-%d %H:%M"), "end": i[-1].strftime("%m-%d %H:%M"),
                     "days": round((i[-1] - i[0]).total_seconds() / 86400, 1),
                     "windows": len(nd.feat), "tier": nd.tier})
    return pd.DataFrame(rows).set_index("segment")


# ---------------------------------------------------------------------------
# 2. Detectors
# ---------------------------------------------------------------------------

def _hour_basis(index: pd.DatetimeIndex, tz: str) -> np.ndarray:
    loc = index.tz_convert(tz)
    w = 2 * np.pi * (loc.hour + loc.minute / 60.0) / 24.0
    return np.column_stack([np.sin(w), np.cos(w)])


class HourResidualizer:
    """Removes the 24 h harmonic from tilt/|acc| (thermal/day-night effect). Fitted on train only, and only
    when train covers >= 3 days and >= 8 distinct hours of day; otherwise it is a no-op."""
    cols = ("dev_u", "dev_v", "norm")

    def fit(self, feat: pd.DataFrame, tz: str):
        self.tz, self.coef = tz, None
        span_days = (feat.index[-1] - feat.index[0]).total_seconds() / 86400
        if span_days < 3 or feat.index.tz_convert(tz).hour.nunique() < 8:
            return self
        B = _hour_basis(feat.index, tz)
        A = np.column_stack([np.ones(len(B)), B])
        self.coef = {c: np.linalg.lstsq(A, feat[c].to_numpy(), rcond=None)[0][1:] for c in self.cols}
        return self

    def transform(self, feat: pd.DataFrame) -> pd.DataFrame:
        out = feat.copy()
        if self.coef is None:
            return out
        B = _hour_basis(feat.index, self.tz)
        for c, k in self.coef.items():
            out[c] = out[c].to_numpy() - B @ k
        return out


def _mad_sigma(x: np.ndarray) -> float:
    return float(1.4826 * np.median(np.abs(x - np.median(x))) + 1e-6)


class NodeModel:
    MODEL_COLS = ["dev_u", "dev_v", "slope_u", "slope_v", "norm", "log_std"]

    def __init__(self, name: str, tier: str, cfg: Cfg):
        self.name, self.tier, self.cfg = name, tier, cfg
        self._lstm = None
        self.lstm_thr = None

    def __getstate__(self):  # keep torch objects out of the pickle
        s = dict(self.__dict__)
        s["_lstm"] = None
        return s

    # -- features -----------------------------------------------------------
    def _prep(self, feat: pd.DataFrame) -> pd.DataFrame:
        f = self.res.transform(feat)
        f["log_std"] = np.log(feat["acc_std_max"].to_numpy() + 1e-4)
        return f

    def _raw_scores(self, feat: pd.DataFrame) -> dict:
        P = self._prep(feat)
        Z = self.scaler.transform(P[self.MODEL_COLS].to_numpy())
        out = {"maha": np.sqrt(self.cov.mahalanobis(Z))}
        if self.iso is not None:
            out["iso"] = -self.iso.score_samples(Z)
        return out

    # -- fit ----------------------------------------------------------------
    def fit(self, nd: NodeData, tr: np.ndarray, va: np.ndarray):
        cfg = self.cfg
        self.res = HourResidualizer().fit(nd.feat[tr], cfg.tz)
        P = self._prep(nd.feat)
        X = P[self.MODEL_COLS].to_numpy()
        self.scaler = RobustScaler().fit(X[tr])
        Ztr = self.scaler.transform(X[tr])

        try:
            if self.tier != "full":
                raise ValueError("light tier uses shrinkage covariance")
            self.cov = MinCovDet(random_state=cfg.seed).fit(Ztr)
        except Exception:
            self.cov = LedoitWolf().fit(Ztr)
        self.iso = (IsolationForest(n_estimators=300, random_state=cfg.seed).fit(Ztr)
                    if self.tier == "full" else None)

        S = self._raw_scores(nd.feat)
        self.thr = {k: float(np.quantile(v[va], cfg.q)) for k, v in S.items()}

        ftr = nd.feat[tr]
        self.norm_med = float(ftr["norm"].median())
        self.std_med = float(ftr["acc_std_max"].median())
        return self

    def attach_lstm(self, pool: "LSTMPool", nd: NodeData, va: np.ndarray):
        self._lstm = pool
        self.lstm_thr = float(np.quantile(pool.error(self.name, nd.seq[va]), self.cfg.q))

    # -- score --------------------------------------------------------------
    def sensor_fault(self, feat: pd.DataFrame) -> np.ndarray:
        flat = feat["acc_std_max"].to_numpy() < max(1e-6, 0.02 * self.std_med)
        off = np.abs(feat["norm"].to_numpy() - self.norm_med) / self.norm_med > self.cfg.norm_tol
        return flat | off

    def score(self, feat: pd.DataFrame, seq: np.ndarray | None = None,
              history: pd.DataFrame | None = None) -> pd.DataFrame:
        S = self._raw_scores(feat)
        if self._lstm is not None and seq is not None:
            S["lstm"] = self._lstm.error(self.name, seq)
        thr = dict(self.thr, **({"lstm": self.lstm_thr} if "lstm" in S else {}))

        shift_input = pd.concat([history, feat]) if history is not None else feat
        shifts = session_shift(shift_input, floor_deg=self.cfg.session_shift_floor_deg)
        shift_deg = feat["session"].map(shifts["shift_deg"])
        shift_alert = feat["session"].map(shifts["alert"]).fillna(False).to_numpy(dtype=bool)

        out = pd.DataFrame(index=feat.index)
        votes = np.zeros(len(feat), dtype=int)
        for k, v in S.items():
            out[k] = v
            out[f"flag_{k}"] = v > thr[k]
            if k != "iso":
                votes += out[f"flag_{k}"].to_numpy()
        out["session_shift_deg"] = shift_deg.to_numpy()
        out["flag_session_shift"] = shift_alert
        votes += shift_alert
        n_det = sum(k != "iso" for k in S) + 1

        sensor = self.sensor_fault(feat)
        out["flag_sensor"] = sensor
        level = pd.Series(votes, index=feat.index).rolling(self.cfg.persist, min_periods=1).min().to_numpy()
        level = np.where(sensor, 0, level).astype(int)
        alarm_v = min(3, n_det)
        warn_v = 2 if alarm_v > 2 else 99
        out["votes"] = votes
        out["status"] = np.select([sensor, level >= alarm_v, level >= warn_v, level >= 1],
                                  ["SENSOR", "ALARM", "WARNING", "WATCH"], "OK")
        return out


# ---------------------------------------------------------------------------
# 3. Optional pooled LSTM autoencoder (short-term signature, per-node normalised)
# ---------------------------------------------------------------------------

if HAVE_TORCH:
    class _LSTMAE(nn.Module):
        def __init__(self, n_feat=3, hidden=32, latent=8):
            super().__init__()
            self.enc = nn.LSTM(n_feat, hidden, batch_first=True)
            self.to_z = nn.Linear(hidden, latent)
            self.from_z = nn.Linear(latent, hidden)
            self.dec = nn.LSTM(hidden, hidden, batch_first=True)
            self.out = nn.Linear(hidden, n_feat)

        def forward(self, x):
            _, (h, _) = self.enc(x)
            z = self.to_z(h[-1])
            d = self.from_z(z).unsqueeze(1).repeat(1, x.size(1), 1)
            return self.out(self.dec(d)[0])


class LSTMPool:
    def __init__(self, cfg: Cfg, epochs=60, batch=128, lr=1e-3, patience=6):
        if not HAVE_TORCH:
            raise RuntimeError("torch is not installed; run without --lstm or `pip install torch`")
        self.cfg, self.epochs, self.batch, self.lr, self.patience = cfg, epochs, batch, lr, patience
        self.sig: dict[str, np.ndarray] = {}

    def fit(self, nodes: dict[str, NodeData]):
        torch.manual_seed(self.cfg.seed)
        Xtr, Xva = [], []
        for name, nd in nodes.items():
            tr, va, _ = nd.masks()
            self.sig[name] = nd.seq[tr].reshape(-1, 3).std(axis=0) + 1e-6
            Xtr.append(nd.seq[tr] / self.sig[name])
            Xva.append(nd.seq[va] / self.sig[name])
        Xtr = torch.tensor(np.concatenate(Xtr), dtype=torch.float32)
        Xva = torch.tensor(np.concatenate(Xva), dtype=torch.float32)

        self.model = _LSTMAE()
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        best, best_state, bad = np.inf, None, 0
        for ep in range(self.epochs):
            self.model.train()
            perm = torch.randperm(len(Xtr))
            for i in range(0, len(Xtr), self.batch):
                xb = Xtr[perm[i:i + self.batch]]
                opt.zero_grad()
                nn.functional.mse_loss(self.model(xb), xb).backward()
                opt.step()
            self.model.eval()
            with torch.no_grad():
                vl = nn.functional.mse_loss(self.model(Xva), Xva).item()
            if vl < best - 1e-5:
                best, bad = vl, 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.model.load_state_dict(best_state)
        logger.info("LSTM-AE trained on %d windows (%d nodes), val MSE %.4f, %d epochs",
                    len(Xtr), len(nodes), best, ep + 1)
        return self

    def error(self, name: str, seq: np.ndarray) -> np.ndarray:
        x = torch.tensor(seq / self.sig[name], dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            return ((self.model(x) - x) ** 2).mean(dim=(1, 2)).numpy()

    def save(self, path):
        torch.save({"state": self.model.state_dict(), "sig": self.sig}, path)


# ---------------------------------------------------------------------------
# 4. Trend report (Theil-Sen, robust) -- slow movement over the whole record
# ---------------------------------------------------------------------------

def trend_report(feat: pd.DataFrame, tz: str) -> dict:
    day = feat.groupby(feat.index.tz_convert(tz).date)[["dev_u", "dev_v"]].median()
    if len(day) < 5:
        return {"note": f"only {len(day)} days with data; need >= 5 for a trend"}
    x = np.arange(len(day), dtype=float)
    out = {}
    for c in ("dev_u", "dev_v"):
        s, b, lo, hi = stats.theilslopes(day[c].to_numpy(), x)
        out[c] = {"deg_per_day": float(s), "ci95": [float(lo), float(hi)], "ci_excludes_zero": bool(lo > 0 or hi < 0)}
    return out


def session_shift(feat: pd.DataFrame | str | Path, ref_n: int = 5,
                  floor_deg: float = 0.11, min_windows: int = 3) -> pd.DataFrame:
    """Compare session tilt with a frozen initial reference and latch threshold crossings."""
    if isinstance(feat, (str, Path)):
        feat = pd.read_csv(feat, index_col=0, parse_dates=True)
    sessions = feat.copy()
    sessions["t"] = sessions.index
    shifts = sessions.groupby("session").agg(
        t=("t", "first"),
        u=("dev_u", "median"),
        v=("dev_v", "median"),
        n=("dev_u", "size"),
    ).sort_values("t")
    shifts["reference_u"] = np.nan
    shifts["reference_v"] = np.nan
    shifts["shift_deg"] = np.nan
    shifts["alert"] = False

    eligible = shifts["n"] >= min_windows
    eligible_ids = shifts.index[eligible]
    baseline_ids = eligible_ids[:ref_n]
    detection_ids = eligible_ids[ref_n:]
    baseline = shifts.loc[baseline_ids]
    if len(baseline) < ref_n:
        return shifts

    reference_u = float(baseline["u"].median())
    reference_v = float(baseline["v"].median())
    shifts["reference_u"] = reference_u
    shifts["reference_v"] = reference_v
    du = shifts.loc[detection_ids, "u"] - reference_u
    dv = shifts.loc[detection_ids, "v"] - reference_v
    shifts.loc[detection_ids, "shift_deg"] = np.hypot(du, dv)

    latched = False
    for session_id in shifts.index:
        if session_id in detection_ids and shifts.loc[session_id, "shift_deg"] > floor_deg:
            latched = True
        shifts.loc[session_id, "alert"] = latched
    return shifts


# ---------------------------------------------------------------------------
# 5. Synthetic-fault evaluation (no labels -> this is how you know the detectors work)
# ---------------------------------------------------------------------------

FAULTS = [("step", 0.3), ("step", 1.0), ("ramp", 1.0), ("stuck", 0.0), ("noise", 5.0)]


def inject(feat: pd.DataFrame, seq: np.ndarray, kind: str, mag: float, onset: int):
    f, s = feat.copy(), seq.copy()
    n = len(f)

    def col(c):
        return f[c].to_numpy().copy()

    if kind == "step":                       # sudden tilt change of `mag` degrees
        v = col("dev_u"); v[onset:] += mag; f["dev_u"] = v
    elif kind == "ramp":                     # tilt creeps by `mag` degrees over the rest of the test period
        v = col("dev_u"); v[onset:] += mag * np.linspace(0, 1, n - onset); f["dev_u"] = v
    elif kind == "stuck":                    # sensor freezes
        for c in ("dev_u", "dev_v", "norm"):
            v = col(c); v[onset:] = v[onset]; f[c] = v
        for c in ("slope_u", "slope_v", "acc_std_max"):
            v = col(c); v[onset:] = 0.0; f[c] = v
        s[onset:] = 0.0
    elif kind == "noise":                    # noise burst, `mag` times louder
        v = col("acc_std_max"); v[onset:] *= mag; f["acc_std_max"] = v
        s[onset:] *= mag
    else:
        raise ValueError(kind)
    return f, s


def evaluate(model: NodeModel, nd: NodeData, te: np.ndarray, cfg: Cfg) -> pd.DataFrame:
    feat, seq = nd.feat[te], nd.seq[te]
    history = nd.feat[~te]
    base = model.score(feat, seq, history=history)
    flag_cols = [c for c in base.columns if c.startswith("flag_")]
    rows = []

    clean = {c[5:]: float(base[c].mean()) for c in flag_cols}
    clean["fused"] = float(base["status"].isin(["WARNING", "ALARM"]).mean())
    clean["any_flag"] = float(base[flag_cols].any(axis=1).mean())
    rows.append({"fault": "none (false-alarm rate)", **clean, "delay_h": np.nan})

    onset = len(feat) // 3
    post_clean = {c[5:]: float(base[c].iloc[onset:].mean()) for c in flag_cols}
    post_clean["fused"] = float(base["status"].iloc[onset:].isin(["WARNING", "ALARM"]).mean())
    post_clean["any_flag"] = float(base[flag_cols].iloc[onset:].any(axis=1).mean())
    rows.append({"fault": "none (same period as faults)", **post_clean, "delay_h": np.nan})
    for kind, mag in FAULTS:
        f2, s2 = inject(feat, seq, kind, mag, onset)
        sc = model.score(f2, s2, history=history)
        post = slice(onset, None)
        r = {"fault": f"{kind} {mag:g}" if kind in ("step", "ramp", "noise") else kind}
        for c in flag_cols:
            r[c[5:]] = float(sc[c].iloc[post].mean())
        fused = sc["status"].isin(["WARNING", "ALARM"]) | (sc["status"] == "SENSOR")
        r["fused"] = float(fused.iloc[post].mean())
        r["any_flag"] = float((sc[flag_cols].any(axis=1)).iloc[post].mean())
        hit = np.flatnonzero(fused.to_numpy()[onset:])
        r["delay_h"] = (float((f2.index[onset + hit[0]] - f2.index[onset]).total_seconds() / 3600)
                        if len(hit) else np.nan)
        rows.append(r)
    return pd.DataFrame(rows).set_index("fault")


# ---------------------------------------------------------------------------
# 5b. Diagnostics: is "healthy" data stationary, and how small a tilt change can this node resolve?
# ---------------------------------------------------------------------------

def diagnose(nd: NodeData) -> pd.DataFrame:
    """Per split: median tilt (a shift between train/val/test means the baseline is not stationary),
    window-to-window scatter, session-to-session scatter, and a rough smallest detectable step (3 sigma)."""
    tr, va, te = nd.masks()
    rows = []
    for split, m in (("train", tr), ("val", va), ("test", te)):
        g = nd.feat[m]
        sm = g.groupby("session")[["dev_u", "dev_v"]].median()
        r = {"split": split, "windows": len(g), "sessions": len(sm),
             "median_u": g["dev_u"].median(), "median_v": g["dev_v"].median(),
             "win_sigma_u": _mad_sigma(g["dev_u"].to_numpy()), "win_sigma_v": _mad_sigma(g["dev_v"].to_numpy()),
             "acc_std_p50": g["acc_std_max"].median(), "acc_std_p95": g["acc_std_max"].quantile(0.95)}
        for c in ("u", "v"):
            r[f"sess_sigma_{c}"] = _mad_sigma(sm[f"dev_{c}"].to_numpy()) if len(sm) >= 3 else np.nan
        r["min_step_deg~3sigma"] = 3 * max(r["win_sigma_u"], r["win_sigma_v"])
        rows.append(r)
    return pd.DataFrame(rows).set_index("split")


# ---------------------------------------------------------------------------
# 6. Orchestration
# ---------------------------------------------------------------------------

def run_all(rfm_df: pd.DataFrame, cfg: Cfg | None = None, out_dir: str | Path = "runs",
            use_lstm: bool = False) -> dict:
    cfg = cfg or Cfg()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.random.seed(cfg.seed)

    d = clean_rfm(rfm_df)
    nodes: dict[str, NodeData] = {}
    skipped = []
    for name, g in d.groupby("sensor_id"):
        segs = prepare_node(name, g, cfg)
        if not segs:
            skipped.append(name)
        for nd in segs:
            nodes[nd.name] = nd
    if not nodes:
        raise RuntimeError("No node has enough clean data to model. Skipped: %s" % skipped)

    pool = None
    if use_lstm:
        full = {k: v for k, v in nodes.items() if v.tier == "full"}
        if not HAVE_TORCH:
            logger.warning("--lstm requested but torch is unavailable; continuing without it.")
        elif not full:
            logger.warning("No node reaches the full tier; skipping LSTM-AE.")
        else:
            pool = LSTMPool(cfg).fit(full)
            pool.save(out / "lstm_ae.pt")

    summary, evals = {}, []
    for name, nd in nodes.items():
        tr, va, te = nd.masks()
        model = NodeModel(name, nd.tier, cfg).fit(nd, tr, va)
        if pool is not None and name in pool.sig:
            model.attach_lstm(pool, nd, va)

        scores = model.score(nd.feat, nd.seq)
        alerts = nd.feat.join(scores)
        shifts = session_shift(nd.feat, floor_deg=cfg.session_shift_floor_deg)
        alerts["split"] = np.select([tr, va], ["train", "val"], "test")
        alerts.to_csv(out / f"{name}_alerts.csv")
        shifts.to_csv(out / f"{name}_session_shift.csv")
        joblib.dump(model, out / f"{name}_model.joblib")

        diag = diagnose(nd)
        diag.to_csv(out / f"{name}_diagnose.csv")
        ev = evaluate(model, nd, te, cfg)
        ev.to_csv(out / f"{name}_eval.csv")
        evals.append(ev.assign(node=name))

        summary[name] = {
            "tier": nd.tier,
            "windows": {"train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum())},
            "detectors": [c[5:] for c in scores.columns if c.startswith("flag_") and c != "flag_sensor"],
            "thresholds": {
                **model.thr,
                **({"lstm": model.lstm_thr} if model.lstm_thr else {}),
                "session_shift_deg": cfg.session_shift_floor_deg,
            },
            "status_counts_test": alerts.loc[te, "status"].value_counts().to_dict(),
            "session_shift_alerts": int(shifts["alert"].sum()),
            "false_alarm_rate_test": ev.loc["none (false-alarm rate)", "fused"],
            "diagnose": diag.round(4).to_dict(orient="index"),
            "trend": trend_report(nd.feat, cfg.tz),
            "ref_gravity_dir": nd.ref.round(4).tolist(),
        }
    summary["_skipped_health_check_only"] = skipped
    summary["_no_valid_rows_after_cleaning"] = sorted(set(rfm_df["sensor_id"].dropna().astype(str)) - set(d["sensor_id"]))
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    pd.concat(evals).to_csv(out / "eval_all_nodes.csv")
    return {"nodes": nodes, "summary": summary, "eval": pd.concat(evals)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="raw InfluxDB export (device,site,time,value)")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--lstm", action="store_true", help="also train the pooled LSTM-AE (needs torch)")
    ap.add_argument("--window", type=int, default=300, help="window length in seconds")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    from intellisenz.preprocess.preprocessing import load_raw_export, preprocess_raw_export

    rfm = preprocess_raw_export(load_raw_export(args.csv))["RFM"]
    res = run_all(rfm, Cfg(window_s=args.window), args.out, use_lstm=args.lstm)
    print("\n=== segments (each has its own baseline) ===")
    print(segment_table(res["nodes"]).to_string())
    pd.set_option("display.width", 200)
    for name, nd in res["nodes"].items():
        print(f"\n=== {name}  [{nd.tier}]  fraction of windows flagged: clean test set (row 1) / after injected fault (other rows) ===")
        print(res["eval"].query("node == @name").drop(columns="node").round(3).to_string())
        print(f"--- {name} diagnostics (degrees) ---")
        print(diagnose(nd).round(3).to_string())
    print(f"\nOutputs written to {args.out}/")


if __name__ == "__main__":
    main()
"""
preprocessing.py

SHM sensor preprocessing (RFL + RFM), extracted from SHM_preprocessing_pipeline.ipynb.

Two layers:

1. Ingest (needs no commissioning data) -- used by load_timescale.py
       load_raw_export()  ->  preprocess_raw_export()  ->  frame_to_records()

2. Signal processing (needs per-node commissioning inputs) -- import what you need
       calibration, rotation, denoising, baseline, confound removal,
       fault gating, feature extraction, run_preprocessing_pipeline()

Step 9 (LSTM-autoencoder / CUSUM / load-event ensemble) is intentionally NOT here:
it is modelling, not preprocessing, and pulls in torch.

RFL packet layouts seen in real data (see parse_payload):
    [sensor_id], device_time | "RTC Error", <9 IMU channels>, <load cells>, [packet_counter]
    load cell v1:  <label>:<raw>,<w>kg                 e.g. 500:-31197,0.00kg
    load cell v2:  <label>T:<raw>,<raw_2>,<w>kg        e.g. 5T:1684,-24221,0.00kg
    legacy placeholder: 11 all-zero fields instead of 9 IMU channels (carries no measurement)
"""

from __future__ import annotations

import argparse
import logging
import math
import re
from datetime import date, datetime, time as dtime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.optimize import least_squares

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RAW_COLUMNS = ["device", "site", "time", "value"]

RFM_FIELDS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z", "mag_x", "mag_y", "mag_z"]
# The first 9 RFL fields look like the same IMU as RFM (acc_z ~ +9.4 m/s^2 = gravity, mag ~ 0.0x).
# INFERRED from the data, not from a datasheet -- confirm with the firmware owner. To go back to
# generic names, use: RFL_FIELDS = [f"rfl_ch{i}" for i in range(1, 10)]
RFL_FIELDS = list(RFM_FIELDS)

# NOTE (audit, Sept 2026): per-sensor sampling is ~10 s (RFM) and ~3 s (RFL), i.e. fs ~ 0.1-0.33 Hz,
# so Nyquist is only ~0.05-0.17 Hz. Anything at "vibration" frequencies (this cutoff, Welch dominant
# frequency, kurtosis/crest factor on single snapshots) is NOT meaningful on this data. Use tilt /
# slow-drift / step-change features instead, unless the firmware can send faster bursts.
STRUCTURE_CUTOFF_HZ = 25.0  # TODO: only meaningful if the device sends vibration-rate data

_SENSOR_ID_RE = re.compile(r"^[A-Za-z]")
_DATE_RE = re.compile(r"^\d{1,2}/\d{1,2}/\d{4}")
_TIME_ONLY_RE = re.compile(r"^\d{1,2}:\d{2}:\d{2}$")
_RTC_ERROR_RE = re.compile(r"^\s*RTC\s*Error\s*$", re.IGNORECASE)
# First token of a load-cell block: "<label>:<int>" e.g. "500:-31197", "7.5:0", "5T:1684", "7.5T:0"
_LC_HEAD_RE = re.compile(r"^\s*(\d+(?:\.\d+)?T?):\s*(-?\d+)\s*$")
_INT_RE = re.compile(r"^\s*-?\d+\s*$")
_KG_RE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*kg\s*$", re.IGNORECASE)


# ===========================================================================
# 0. Load + parse the raw export
# ===========================================================================

def load_raw_export(path: str) -> pd.DataFrame:
    """Read the headerless InfluxDB export: device, site, time, value.

    A header row (if present) is dropped.
    """
    df = pd.read_csv(path, header=None, names=RAW_COLUMNS, quotechar='"', engine="python")
    if len(df) and str(df.iloc[0]["time"]).strip().lower() == "time":
        df = df.iloc[1:].reset_index(drop=True)
    return df


def _safe_float(tok):
    try:
        return float(tok)
    except (TypeError, ValueError):
        return np.nan


def _parse_load_cells(tail):
    """Parse the load-cell block of an RFL packet.

    Returns (cells, packet_counter, leftover_tokens). Each cell is
    {label, raw, raw_2 (v2 only, else None), weight_kg_reported}. `label` is the text before the
    colon ("500", "7.5", "5T", "7.5T"); it is kept as a string because its meaning is unconfirmed.
    Anything that is neither a load cell nor a single trailing integer counter is returned as
    leftover so the caller can flag the packet instead of silently dropping data.
    """
    cells, i = [], 0
    while i < len(tail):
        m = _LC_HEAD_RE.match(tail[i])
        if not m:
            break
        label, raw = m.group(1), int(m.group(2))
        if i + 2 < len(tail) and _INT_RE.match(tail[i + 1]) and _KG_RE.match(tail[i + 2]):
            cells.append({"label": label, "raw": raw, "raw_2": int(tail[i + 1]),
                          "weight_kg_reported": float(_KG_RE.match(tail[i + 2]).group(1))})
            i += 3
        elif i + 1 < len(tail) and _KG_RE.match(tail[i + 1]):
            cells.append({"label": label, "raw": raw, "raw_2": None,
                          "weight_kg_reported": float(_KG_RE.match(tail[i + 1]).group(1))})
            i += 2
        else:
            break
    rest = tail[i:]
    counter = None
    if len(rest) == 1 and _INT_RE.match(rest[0]):
        counter = int(rest[0])
        rest = []
    return cells, counter, rest


def parse_payload(raw_value, device_type):
    """Parse one packed `value` string into a flat dict.

    Pieces are detected by shape, not position, because real packets vary:
      * `sensor_id` may be missing.
      * device date/time may be one token, two tokens, or the literal text "RTC Error".
      * RFL load cells come in two layouts (v1 pairs, v2 triplets) and the packet counter is optional.
      * a legacy RFL packet has 11 all-zero fields instead of 9 channels.

    Output flags (all bool):
      parse_error          content we could not explain (leftover tokens, wrong field count)
      rtc_error            device reported "RTC Error" instead of a timestamp
      channels_all_zero    the 9 IMU channels are all zero / missing (dead IMU or placeholder packet);
                           physically impossible for an accelerometer, so treat the channels as invalid
      load_cells_all_zero  (RFL) every load-cell value is zero -> load cell not connected/responding
    Channels are left as NaN for legacy placeholder packets.
    """
    if not isinstance(raw_value, str):
        return {"parse_error": True}
    tokens = [t.strip() for t in raw_value.strip().strip('"').split(",")]
    if not tokens or tokens == [""]:
        return {"parse_error": True}

    out = {"sensor_id": None, "device_time_raw": None, "rtc_error": False, "parse_error": False}
    idx = 0

    # optional sensor_id, and/or an "RTC Error" marker in place of the timestamp
    if _RTC_ERROR_RE.match(tokens[0]):
        out["rtc_error"] = True
        idx = 1
    elif _SENSOR_ID_RE.match(tokens[0]):
        out["sensor_id"] = tokens[0]
        idx = 1
        if idx < len(tokens) and _RTC_ERROR_RE.match(tokens[idx]):
            out["rtc_error"] = True
            idx += 1

    if not out["rtc_error"] and idx < len(tokens) and _DATE_RE.match(tokens[idx]):
        if " " in tokens[idx]:
            out["device_time_raw"] = tokens[idx]
            idx += 1
        elif idx + 1 < len(tokens) and _TIME_ONLY_RE.match(tokens[idx + 1]):
            out["device_time_raw"] = f"{tokens[idx]} {tokens[idx + 1]}"
            idx += 2
        else:
            out["device_time_raw"] = tokens[idx]  # date with no matching time token
            idx += 1

    body = tokens[idx:]
    if device_type == "RFM":
        front, tail = body[:9], []  # RFM: 9 channels, anything after is ignored (as before)
    else:
        lc_at = next((i for i, t in enumerate(body) if _LC_HEAD_RE.match(t)), None)
        front = body if lc_at is None else body[:lc_at]
        tail = [] if lc_at is None else [t for t in body[lc_at:] if t != ""]
        while front and front[-1] == "":
            front.pop()

    values = np.array([_safe_float(t) for t in front], dtype=float)
    out["nan_field_count"] = int(np.isnan(values).sum())
    zero_like = bool(len(values)) and bool(np.all((values == 0) | np.isnan(values)))
    field_names = RFM_FIELDS if device_type == "RFM" else RFL_FIELDS

    if len(values) == 9:
        out.update(dict(zip(field_names, values.tolist())))
        out["channels_all_zero"] = zero_like
    elif device_type == "RFL" and len(values) > 9 and zero_like:
        out["channels_all_zero"] = True  # legacy 11-field placeholder: no measurement to keep
    else:
        out["parse_error"] = True
        out["unparsed_front"] = front

    if device_type == "RFL":
        cells, counter, leftover = _parse_load_cells(tail)
        out["load_cells"] = cells
        out["packet_counter"] = counter
        out["load_cells_all_zero"] = bool(cells) and all(
            c["raw"] == 0 and c["raw_2"] in (None, 0) and c["weight_kg_reported"] == 0 for c in cells
        )
        if leftover:
            out["parse_error"] = True
            out["unparsed_tail"] = leftover
    return out


def build_device_frame(df_raw, device_type):
    sub = df_raw[df_raw["device"] == device_type].copy()
    parsed = sub["value"].apply(lambda v: parse_payload(v, device_type))
    parsed_df = pd.DataFrame(list(parsed))
    return pd.concat(
        [sub[["device", "site", "time"]].reset_index(drop=True), parsed_df.reset_index(drop=True)],
        axis=1,
    )


# ===========================================================================
# Step 1 -- Time alignment (server time is ground truth)
# ===========================================================================

def add_time_columns(df, device_type=None):
    """`server_time` (InfluxDB timestamp) is authoritative. `device_time` is only used to
    flag RTC resets, never for ordering. `device_type` is kept for API compatibility."""
    df = df.copy()
    if df.empty and "device_time_raw" not in df.columns:
        df["device_time_raw"] = pd.Series(index=df.index, dtype="object")
    # format="ISO8601" so rows with and without fractional seconds all parse; pandas otherwise
    # infers the format from the first row and turns the rest into NaT.
    df["server_time"] = pd.to_datetime(df["time"], utc=True, errors="coerce", format="ISO8601")
    df["server_date"] = df["server_time"].dt.date
    df["server_time_of_day"] = df["server_time"].dt.time

    df["device_time"] = pd.to_datetime(
        df["device_time_raw"], format="%d/%m/%Y %H:%M:%S", errors="coerce"
    )
    df["device_date"] = df["device_time"].dt.date
    df["device_time_of_day"] = df["device_time"].dt.time

    df["device_clock_bad"] = df["device_time"].isna() | (df["device_time"] < pd.Timestamp("2015-01-01"))
    return df.sort_values("server_time").reset_index(drop=True)


def resample_fixed_rate(df, node_col, time_col, value_cols, freq, slow_channels=None,
                        max_interp_gap="2s"):
    """Resample each node's stream to a strict fixed rate.

    `slow_channels` (tilt/load) may be linearly interpolated over short gaps. Everything
    else (vibration) is left NaN across a gap; that window must be marked invalid later.
    """
    slow_channels = slow_channels or []
    out = []
    for node, g in df.groupby(node_col):
        g = g.set_index(time_col).sort_index()
        g = g[~g.index.duplicated(keep="first")]
        idx = pd.date_range(g.index.min(), g.index.max(), freq=freq)
        r = g[value_cols].reindex(idx, method="nearest", tolerance=pd.Timedelta(freq) / 2)
        gap_mask = r[value_cols].isna().any(axis=1)
        for c in value_cols:
            if c in slow_channels:
                r[c] = r[c].interpolate(limit=int(pd.Timedelta(max_interp_gap) / pd.Timedelta(freq)))
        r["window_gap_flag"] = gap_mask
        r[node_col] = node
        r.index.name = time_col
        out.append(r.reset_index())
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def align_streams_by_transient(series_a, series_b, fs):
    """Lag (seconds) to shift series_b onto series_a, via cross-correlation of a shared transient."""
    a = np.nan_to_num(series_a.values - np.nanmean(series_a.values))
    b = np.nan_to_num(series_b.values - np.nanmean(series_b.values))
    corr = signal.correlate(a, b, mode="full")
    return (corr.argmax() - (len(b) - 1)) / fs


# ===========================================================================
# Step 2 -- Calibration
# ===========================================================================

def six_position_calibration(static_captures, g=9.80665):
    """Per-axis bias/scale such that ||(raw - b) * s|| ~= g in every stationary capture."""
    all_samples = np.vstack(list(static_captures.values()))

    def residuals(params):
        bx, by, bz, sx, sy, sz = params
        corrected = (all_samples - [bx, by, bz]) * [sx, sy, sz]
        return np.linalg.norm(corrected, axis=1) - g

    result = least_squares(residuals, [0, 0, 0, 1, 1, 1])
    bx, by, bz, sx, sy, sz = result.x
    return {
        "bias": np.array([bx, by, bz]),
        "scale": np.array([sx, sy, sz]),
        "rmse": np.sqrt(np.mean(result.fun ** 2)),
    }


def estimate_gyro_bias(stationary_gyro_window):
    return np.nanmean(stationary_gyro_window, axis=0)


def apply_accel_calibration(df, cols, bias, scale):
    df = df.copy()
    for i, c in enumerate(cols):
        df[c + "_cal"] = (df[c] - bias[i]) * scale[i]
    return df


def apply_gyro_debias(df, cols, bias):
    df = df.copy()
    for i, c in enumerate(cols):
        df[c + "_cal"] = df[c] - bias[i]
    return df


def load_cell_calibrate(raw_adc, tare_raw, scale_factor, temp=None, ref_temp=None, temp_coeff=None):
    """weight_kg = (raw - tare) * scale, minus a linear temperature correction if available."""
    weight = (raw_adc - tare_raw) * scale_factor
    if temp is not None and ref_temp is not None and temp_coeff is not None:
        weight = weight - temp_coeff * (temp - ref_temp)
    return weight


# ===========================================================================
# Step 3 -- Canonical frame rotation
# ===========================================================================

def rotation_from_gravity(stationary_accel_mean, g_ref=np.array([0, 0, 1.0])):
    """Rotation R such that R @ normalize(stationary_accel_mean) ~= g_ref (Rodrigues)."""
    v = stationary_accel_mean / np.linalg.norm(stationary_accel_mean)
    g = g_ref / np.linalg.norm(g_ref)
    axis = np.cross(v, g)
    s = np.linalg.norm(axis)
    c = np.dot(v, g)
    if s < 1e-8:
        return np.eye(3) if c > 0 else -np.eye(3)
    axis = axis / s
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + K * s + K @ K * (1 - c)


def apply_rotation(df, cols, R, out_prefix="rot_"):
    df = df.copy()
    rotated = df[cols].to_numpy() @ R.T
    for i, c in enumerate(cols):
        df[out_prefix + c] = rotated[:, i]
    return df


def static_dynamic_split(series, fs, cutoff_hz=0.3, order=4):
    """Static (gravity/tilt) = low-pass; dynamic (vibration) = residual."""
    b, a = signal.butter(order, cutoff_hz / (fs / 2), btype="low")
    static = signal.filtfilt(b, a, series.ffill().bfill())
    dynamic = series.values - static
    return pd.Series(static, index=series.index), pd.Series(dynamic, index=series.index)


# ===========================================================================
# Step 4 -- Denoising
# ===========================================================================

def lowpass_filter(series, fs, cutoff_hz=STRUCTURE_CUTOFF_HZ, order=4):
    nyq = fs / 2
    cutoff = min(cutoff_hz, nyq * 0.99)
    b, a = signal.butter(order, cutoff / nyq, btype="low")
    return pd.Series(signal.filtfilt(b, a, series.ffill().bfill()), index=series.index)


def hampel_filter(series, window_size=7, n_sigmas=3):
    """Median-based single-sample glitch removal."""
    k = 1.4826
    rolling_median = series.rolling(window_size, center=True).median()
    diff = (series - rolling_median).abs()
    mad = diff.rolling(window_size, center=True).median()
    outlier_mask = diff > n_sigmas * k * mad
    cleaned = series.copy()
    cleaned[outlier_mask] = rolling_median[outlier_mask]
    return cleaned, outlier_mask


def denoise_columns(df, cols, fs, hampel_window=7, hampel_sigmas=3):
    """Adds `<col>_filt` and `<col>_glitch`; the raw column is never overwritten."""
    df = df.copy()
    for c in cols:
        clean, glitch_mask = hampel_filter(df[c], hampel_window, hampel_sigmas)
        df[c + "_filt"] = lowpass_filter(clean, fs)
        df[c + "_glitch"] = glitch_mask
    return df


# ===========================================================================
# Step 5 -- Commissioning baseline
# ===========================================================================

def build_node_baseline(node_df, feature_cols, fs, welch_nperseg=256):
    """node_df: ONE node's data, restricted to its healthy commissioning window."""
    baseline = {"mean": {}, "std": {}, "psd_ref": {}, "dominant_freq": {}}
    for c in feature_cols:
        vals = node_df[c].dropna()
        baseline["mean"][c] = float(vals.mean())
        baseline["std"][c] = float(vals.std())
        if len(vals) >= welch_nperseg:
            f, pxx = signal.welch(vals.to_numpy(), fs=fs, nperseg=welch_nperseg)
            baseline["psd_ref"][c] = {"f": f.tolist(), "pxx": pxx.tolist()}
            baseline["dominant_freq"][c] = float(f[np.argmax(pxx)])
    return baseline


# ===========================================================================
# Step 6 -- Confound removal
# ===========================================================================

def harmonic_time_features(timestamps, periods_hours=(24, 24 * 365.25)):
    """Fourier terms for time-of-day/seasonal periodicity (confound proxy without a temp sensor)."""
    t = (timestamps - timestamps.min()).total_seconds() / 3600.0
    feats = {}
    for p in periods_hours:
        feats[f"sin_{p:g}h"] = np.sin(2 * np.pi * t / p)
        feats[f"cos_{p:g}h"] = np.cos(2 * np.pi * t / p)
    return pd.DataFrame(feats, index=timestamps.index if hasattr(timestamps, "index") else None)


def regress_out_confound(feature, confound_df, healthy_mask):
    """OLS fit on the healthy period only; residual computed over the whole series."""
    X_fit = confound_df.loc[healthy_mask].to_numpy()
    y_fit = feature.loc[healthy_mask].to_numpy()
    coef, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(X_fit)), X_fit]), y_fit, rcond=None)
    pred = np.column_stack([np.ones(len(confound_df)), confound_df.to_numpy()]) @ coef
    return pd.Series(feature.to_numpy() - pred, index=feature.index), coef


def cointegration_residual(y, x):
    """Engle-Granger step 1: residual of y ~ alpha + beta * x."""
    coef, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), y, rcond=None)
    alpha, beta = coef
    return y - (alpha + beta * x), {"alpha": alpha, "beta": beta}


try:
    from statsmodels.tsa.stattools import coint as _sm_coint

    def coint_test(y, x):
        """(test_stat, p_value, crit_values). p < 0.05 -> residual is a valid stationary index."""
        return _sm_coint(y, x)
except ImportError:  # statsmodels is optional
    def coint_test(y, x):
        logger.warning("statsmodels not installed; skipping formal cointegration test.")
        return None


# ===========================================================================
# Step 7 -- Sensor-fault gating
# ===========================================================================

def detect_flatline(series, window=20, tol=1e-6):
    return series.rolling(window).std() < tol


def detect_saturation(series, sensor_min, sensor_max, margin=0.01):
    span = sensor_max - sensor_min
    return (series <= sensor_min + margin * span) | (series >= sensor_max - margin * span)


def detect_packet_loss(server_time, expected_interval_s, tolerance=2.0):
    return server_time.diff().dt.total_seconds() > expected_interval_s * tolerance


def cross_axis_coherence(df, cols, window=50, corr_threshold=0.3):
    """True = low mean pairwise correlation -> likely an isolated single-sensor issue."""
    roll_corr = df[cols].rolling(window).corr()
    n = len(cols)
    mean_abs_corr = roll_corr.abs().groupby(level=0).apply(
        lambda m: (m.to_numpy().sum() - n) / (n ** 2 - n) if n > 1 else np.nan
    )
    return mean_abs_corr < corr_threshold


def build_fault_flags(df, value_cols, server_time_col, sensor_ranges, expected_interval_s):
    flags = pd.DataFrame(index=df.index)
    for c in value_cols:
        flags[c + "_flatline"] = detect_flatline(df[c])
        if c in sensor_ranges:
            lo, hi = sensor_ranges[c]
            flags[c + "_saturated"] = detect_saturation(df[c], lo, hi)
    flags["packet_loss"] = detect_packet_loss(df[server_time_col], expected_interval_s)
    if "device_clock_bad" in df.columns:
        flags["rtc_reset"] = df["device_clock_bad"]
    if "rtc_error" in df.columns:
        flags["rtc_error"] = df["rtc_error"].fillna(False).astype(bool)
    if "channels_all_zero" in df.columns:
        flags["channels_all_zero"] = df["channels_all_zero"].fillna(False).astype(bool)
    flags["isolated_sensor_deviation"] = cross_axis_coherence(df, value_cols)
    flags["any_fault"] = flags.any(axis=1)
    return flags


# ===========================================================================
# Step 8 -- Feature extraction
# ===========================================================================

def time_domain_features(window):
    w = window.dropna()
    if len(w) < 2:
        return {k: np.nan for k in ["rms", "peak", "crest_factor", "kurtosis", "skewness"]}
    rms = np.sqrt(np.mean(w ** 2))
    peak = np.max(np.abs(w))
    return {
        "rms": rms,
        "peak": peak,
        "crest_factor": peak / rms if rms > 0 else np.nan,
        "kurtosis": stats.kurtosis(w),
        "skewness": stats.skew(w),
    }


def freq_domain_features(window, fs, nperseg=256):
    w = window.dropna()
    if len(w) < nperseg:
        return {"dominant_freq": np.nan, "dominant_amp": np.nan}
    f, pxx = signal.welch(w.to_numpy(), fs=fs, nperseg=nperseg)
    idx = np.argmax(pxx)
    return {"dominant_freq": f[idx], "dominant_amp": pxx[idx]}


def windowed_features(series, fs, window_s, overlap=0.5, freq_domain=True):
    win_len = int(window_s * fs)
    step = max(1, int(win_len * (1 - overlap)))
    rows = []
    for start in range(0, max(1, len(series) - win_len + 1), step):
        w = series.iloc[start:start + win_len]
        feats = time_domain_features(w)
        if freq_domain:
            feats.update(freq_domain_features(w, fs))
        feats["window_start"] = w.index[0] if len(w) else None
        rows.append(feats)
    return pd.DataFrame(rows)


def extract_rfm_features(node_df, fs, window_s=10, overlap=0.5,
                         axis_cols=("rot_acc_x", "rot_acc_y", "rot_acc_z")):
    """Dynamic/vibration modality: windowed time + frequency features per axis."""
    return {c: windowed_features(node_df[c], fs, window_s, overlap, freq_domain=True)
            for c in axis_cols if c in node_df.columns}


def extract_rfl_features(node_df, fs, window_s=60, overlap=0.5,
                         quasi_static_cols=("rot_acc_x", "rot_acc_y", "rot_acc_z")):
    """Quasi-static modality: longer windows, time-domain only."""
    return {c: windowed_features(node_df[c], fs, window_s, overlap, freq_domain=False)
            for c in quasi_static_cols if c in node_df.columns}


def fuse_decision(rfl_anomaly_score, rfm_anomaly_score, mode="max"):
    """Decision-level fusion; never concatenate RFL/RFM feature vectors."""
    if mode == "max":
        return max(rfl_anomaly_score, rfm_anomaly_score)
    if mode == "weighted":
        return 0.5 * rfl_anomaly_score + 0.5 * rfm_anomaly_score
    raise ValueError(mode)


# ===========================================================================
# End-to-end wrapper (per node; needs commissioning inputs)
# ===========================================================================

def run_preprocessing_pipeline(node_df, device_type, fs, node_id,
                               accel_cols=None, calib=None, rotation=None,
                               sensor_ranges=None, expected_interval_s=None,
                               healthy_mask=None, confound_series=None):
    """Runs Steps 2-8 for one node's already time-sorted DataFrame.

    Returns (processed_df, fault_flags, baseline, features).
    """
    df = node_df.copy()

    if accel_cols and calib is not None:
        df = apply_accel_calibration(df, accel_cols, calib["bias"], calib["scale"])
        cal_cols = [c + "_cal" for c in accel_cols]
    else:
        cal_cols = accel_cols or []

    if rotation is not None and cal_cols:
        df = apply_rotation(df, cal_cols, rotation)
        rot_cols = ["rot_" + c for c in cal_cols]
        for c in rot_cols:
            static, dynamic = static_dynamic_split(df[c], fs)
            df[c + "_static"] = static
            df[c + "_dynamic"] = dynamic
    else:
        rot_cols = cal_cols

    if rot_cols:
        df = denoise_columns(df, rot_cols, fs)

    if rot_cols and healthy_mask is not None and confound_series is not None:
        resid, _ = regress_out_confound(
            df[rot_cols[0]], pd.DataFrame({"confound": confound_series}), healthy_mask
        )
        df[rot_cols[0] + "_confound_resid"] = resid

    fault_flags = None
    if sensor_ranges and expected_interval_s and rot_cols:
        fault_flags = build_fault_flags(df, rot_cols, "server_time", sensor_ranges, expected_interval_s)

    baseline = None
    if healthy_mask is not None and rot_cols:
        baseline = build_node_baseline(df.loc[healthy_mask], rot_cols, fs)

    features = None
    if rot_cols:
        if device_type == "RFM":
            features = extract_rfm_features(df, fs, axis_cols=tuple(rot_cols))
        else:
            features = extract_rfl_features(df, fs, quasi_static_cols=tuple(rot_cols))

    return df, fault_flags, baseline, features


# ===========================================================================
# Ingest entry points (used by load_timescale.py)
# ===========================================================================

def preprocess_raw_export(df_raw: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Parse the raw export and apply Step 1 time handling. Needs no commissioning data.

    Returns {"RFL": df, "RFM": df}, each sorted by server_time.
    """
    frames = {}
    for device_type in ("RFL", "RFM"):
        frames[device_type] = add_time_columns(build_device_frame(df_raw, device_type), device_type)
    unknown = set(df_raw["device"].dropna().unique()) - set(frames)
    if unknown:
        logger.warning("Ignoring unknown device type(s): %s", sorted(unknown))
    logger.info("Parsed RFL rows: %d | RFM rows: %d", len(frames["RFL"]), len(frames["RFM"]))
    return frames


def _json_safe(value):
    """Make a value JSONB-safe. Postgres rejects NaN/Infinity, so those become null."""
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if (math.isnan(value) or math.isinf(value)) else float(value)
    if isinstance(value, (pd.Timestamp, datetime, date, dtime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


_RECORD_DROP = {"time", "device", "server_time"}


def frame_to_records(df: pd.DataFrame) -> list[tuple[datetime, str, dict]]:
    """Convert a preprocessed device frame into (time, measurement, data) tuples.

    time        -> server_time (authoritative clock)
    measurement -> device type (RFL / RFM)
    data        -> everything else, JSON-safe
    Rows with an unparseable server_time are skipped (the column is NOT NULL).
    """
    records, skipped = [], 0
    for row in df.to_dict(orient="records"):
        ts = row.get("server_time")
        if ts is None or pd.isna(ts):
            skipped += 1
            continue
        data = {k: _json_safe(v) for k, v in row.items() if k not in _RECORD_DROP}
        records.append((ts.to_pydatetime(), row["device"], data))
    if skipped:
        logger.warning("Skipped %d row(s) with unparseable server time.", skipped)
    return records


def main() -> None:
    """Parse a raw export and write one parquet file per supported device type."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", help="raw InfluxDB export: device, site, time, value")
    parser.add_argument(
        "--save-parquet",
        default="parsed_output",
        help="directory for rfm.parquet and rfl.parquet (default: parsed_output)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    frames = preprocess_raw_export(load_raw_export(args.csv))
    output_dir = Path(args.save_parquet)
    output_dir.mkdir(parents=True, exist_ok=True)
    for device_type, frame in frames.items():
        path = output_dir / f"{device_type.lower()}.parquet"
        frame.to_parquet(path, index=False)
        logger.info("Wrote %d %s rows to %s", len(frame), device_type, path)


if __name__ == "__main__":
    main()

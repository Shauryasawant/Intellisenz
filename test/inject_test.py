"""inject_test.py -- add fake tilt steps to REAL raw accelerometer data and see if the level detector notices.

Why raw level: the fake change goes through cleaning, windowing and the session logic exactly like a real one.
A step of `theta` degrees is made by rotating every accelerometer vector after the onset time about an axis
perpendicular to gravity, which tilts the apparent gravity direction by `theta`.

Usage (from the project folder):
    .venv/bin/python inject_test.py --csv influx_data.csv --sensor RFM_0002 \
        --start "2026-09-17 08:30" --end "2026-09-19 20:40"        # times in UTC; pick a CLEAN stretch

Read the CONTROL row first: with theta = 0 there must be no alarm. If the control alarms, your chosen
stretch already contains a real step or handling, so pick a cleaner one.
"""
import argparse
import logging

import numpy as np
import pandas as pd

from intellisenz.models.SHM_modelv2 import (
    ACC, Cfg, _basis, _seconds, build_segment, clean_rfm, run_session_detector, session_table, split_segments,
)
from intellisenz.preprocess.preprocessing import load_raw_export, preprocess_raw_export


def rotate(acc, k, theta_deg):
    """Rodrigues rotation of every row of `acc` about unit axis `k` by theta_deg (scalar or per-row array)."""
    th = np.radians(np.broadcast_to(theta_deg, (len(acc),)))[:, None]
    return (acc * np.cos(th)
            + np.cross(k, acc) * np.sin(th)
            + k[None, :] * (acc @ k)[:, None] * (1 - np.cos(th)))


def run_case(gs, cfg, theta, onset_frac, ramp_h):
    """Inject, run the detector, return (detected, delay_hours, note)."""
    t = _seconds(gs["server_time"])
    onset = t[0] + onset_frac * (t[-1] - t[0])
    acc = gs[ACC].to_numpy(float)
    unit = acc / np.linalg.norm(acc, axis=1)[:, None]
    ref = np.median(unit, axis=0)
    ref /= np.linalg.norm(ref)
    k, _ = _basis(ref)                                   # axis perpendicular to gravity

    ramp_s = max(ramp_h * 3600.0, 1e-9)
    theta_t = theta * np.clip((t - onset) / ramp_s, 0.0, 1.0)   # 0 before onset, full theta after (or ramps up)
    g2 = gs.copy()
    g2[ACC] = rotate(acc, k, theta_t)

    built = build_segment(g2, cfg)
    if built is None:
        return False, np.nan, "too little data"
    F, _ = built
    S = run_session_detector(session_table(F), cfg, rebaseline=False)
    after = int(((S["t_h"] > onset / 3600.0) & (S["n"] >= cfg.min_win)).sum())   # sessions that can confirm
    before = int(((S["t_h"] <= onset / 3600.0) & (S["n"] >= cfg.min_win)).sum())  # usable sessions before the step
    base = S[S["state"] == "BASE"]
    if len(base):
        armed_h = base["t_h"].max()
        arm_txt = (f"step BEFORE arming (reference built from data that already contains it)"
                   if onset / 3600.0 <= armed_h else "step after arming")
    else:
        arm_txt = "never armed"
    if not (S["state"] == "ALARM").any():
        if (S["state"] == "PREARM_SHIFT").any() and len(base):
            found_t = base["t"].max()                       # the reference completes here; that is when we can tell
            delay = (found_t - pd.Timestamp(onset, unit="s", tz="UTC")).total_seconds() / 3600.0
            return True, delay, "PREARM_SHIFT (step found while arming; reported when the reference completed)"
        return False, np.nan, f"no alarm; {arm_txt}; usable sessions before/after onset: {before}/{after}"
    # the alarm is only known when the confirming (persist-th) session has been seen
    confirm_t = S.loc[S["shift_deg"].last_valid_index(), "t"]
    onset_ts = pd.Timestamp(onset, unit="s", tz="UTC")
    return True, (confirm_t - onset_ts).total_seconds() / 3600.0, f"ALARM; {arm_txt}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--sensor", required=True)
    ap.add_argument("--start", required=True, help="UTC, e.g. '2026-09-17 08:30'")
    ap.add_argument("--end", required=True, help="UTC")
    ap.add_argument("--thetas", type=float, nargs="*", default=[0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0])
    ap.add_argument("--onsets", type=float, nargs="*", default=[0.5, 0.6, 0.7], help="onset positions (fraction of the stretch)")
    ap.add_argument("--ramp-h", type=float, default=0.0, help="0 = sudden step; e.g. 24 = ramp up over 24 h")
    ap.add_argument("--floor", type=float, help="alarm floor in degrees (default 0.12)")
    ap.add_argument("--ref-min-h", type=float, help="min time span of the reference sessions (default 12)")
    ap.add_argument("--ref-min-sessions", type=int, help="min number of reference sessions (default 4)")
    ap.add_argument("--agree-deg", type=float, help="how closely reference sessions must agree (default 0.10)")
    ap.add_argument("--detail", action="store_true", help="print every single case with the reason for a miss")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    cfg = Cfg()
    for arg, field in [("floor", "floor_deg"), ("ref_min_h", "ref_min_h"),
                       ("ref_min_sessions", "ref_min_sessions"), ("agree_deg", "ref_agree_deg")]:
        if getattr(args, arg) is not None:
            setattr(cfg, field, getattr(args, arg))
    cfg.prearm_check = True                       # count 'step found while arming' as a detection
    d = clean_rfm(preprocess_raw_export(load_raw_export(args.csv))["RFM"])
    g = d[d["sensor_id"] == args.sensor]
    g = g[(g["server_time"] >= pd.Timestamp(args.start, tz="UTC")) & (g["server_time"] <= pd.Timestamp(args.end, tz="UTC"))]
    g = split_segments(g.reset_index(drop=True), cfg)
    if g.empty:
        raise SystemExit("no usable rows in that time range")
    biggest = g.groupby("segment").size().idxmax()
    gs = g[g["segment"] == biggest].reset_index(drop=True)
    span_h = (gs["server_time"].iloc[-1] - gs["server_time"].iloc[0]).total_seconds() / 3600
    print(f"{args.sensor}: {len(gs)} rows, {span_h:.0f} h in the chosen orientation segment "
          f"(alarm floor {cfg.floor_deg} deg, confirm after {cfg.persist_sessions} sessions)")

    built0 = build_segment(gs, cfg)
    if built0 is not None:
        S0 = session_table(built0[0])
        gaps_h = np.diff(S0["t_h"].to_numpy())
        print(f"sessions: {len(S0)}, median {S0['n'].median():.0f} windows each, "
              f"median spacing between session starts {np.median(gaps_h):.1f} h (max {gaps_h.max():.1f} h)")

        S0d = run_session_detector(S0, cfg, rebaseline=False)
        b0 = S0d[S0d["state"] == "BASE"]
        if len(b0):
            armed = b0["t"].max()
            hrs = (armed - gs["server_time"].iloc[0]).total_seconds() / 3600
            print(f"detector armed (reference complete) at {armed.tz_convert(cfg.tz).strftime('%m-%d %H:%M')} IST, "
                  f"{hrs:.0f} h after the stretch starts; a step before that is NOT monitored")
        else:
            print("detector never armed on this stretch")

    rows, detail = [], []
    for th in args.thetas:
        res = [run_case(gs, cfg, th, o, args.ramp_h) for o in args.onsets]
        for o, r in zip(args.onsets, res):
            onset_ts = gs["server_time"].iloc[0] + o * (gs["server_time"].iloc[-1] - gs["server_time"].iloc[0])
            detail.append({"injected_deg": th, "onset_frac": o,
                           "onset_IST": onset_ts.tz_convert(cfg.tz).strftime("%m-%d %H:%M"),
                           "detected": r[0], "delay_h": None if np.isnan(r[1]) else round(r[1], 1), "note": r[2]})
        det = [r[0] for r in res]
        delays = [r[1] for r in res if r[0]]
        rows.append({"injected_deg": th, "case": "CONTROL" if th == 0 else ("ramp" if args.ramp_h else "step"),
                     "detected": f"{sum(det)}/{len(det)}",
                     "median_delay_h": round(float(np.median(delays)), 1) if delays else np.nan})
    print(pd.DataFrame(rows).to_string(index=False))
    if args.detail:
        pd.set_option("display.width", 250, "display.max_colwidth", 90)
        print("\n" + pd.DataFrame(detail).to_string(index=False))
    print("\ndelay = hours from the fake step to the confirming session (an alarm needs "
          f"{cfg.persist_sessions} consecutive hit sessions, so it depends on how often the sensor reports).")


if __name__ == "__main__":
    main()
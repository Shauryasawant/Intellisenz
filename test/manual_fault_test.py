"""
manual_fault_test.py -- inject ONE fault of your choice into real sensor data and see if SHM_modelv3 catches it.

USAGE
    python manual_fault_test.py --csv influx_data.csv --sensor RFM_0002 --segment 4 \
        --kind step --mag 0.3 --onset_frac 0.6

--kind:   step | ramp | stuck | noise
--mag:    step/ramp -> degrees total shift; noise -> multiplier (e.g. 5 = 5x normal scatter)
--hours:  only used for ramp -- how many hours the ramp takes to reach --mag
--onset_frac: where in the segment (0-1) to start the fault, e.g. 0.6 = 60% of the way through
"""
import argparse
import numpy as np
import pandas as pd

from intellisenz.preprocess.preprocessing import load_raw_export, preprocess_raw_export
from intellisenz.models import SHM_modelv2 as v2
from intellisenz.models import SHM_modelv3 as v3

ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True)
ap.add_argument("--sensor", required=True, help="e.g. RFM_0002")
ap.add_argument("--segment", type=int, required=True, help="1-based, matches the _s<N> in segment names")
ap.add_argument("--kind", choices=["step", "ramp", "stuck", "noise"], required=True)
ap.add_argument("--mag", type=float, default=0.3)
ap.add_argument("--hours", type=float, default=48.0)
ap.add_argument("--onset_frac", type=float, default=None, help="0-1, where in the segment to start the fault")
ap.add_argument("--onset_time", type=str, default=None,
                help="exact timestamp to start the fault, e.g. '2026-09-15 00:00' (parsed as UTC unless you add a tz offset). "
                     "Overrides --onset_frac if given.")
ap.add_argument("--list_sessions", action="store_true",
                help="just print the real (uninjected) session table for this segment and exit -- use this first "
                     "to pick a clean onset time/window that is BASE/OK and before any real EVENT/ALARM/DRIFT")
args = ap.parse_args()
if args.onset_frac is None and args.onset_time is None:
    args.onset_frac = 0.6

cfg = v3.Cfg3()
c2 = cfg.v2()

rfm = preprocess_raw_export(load_raw_export(args.csv))["RFM"]
d = v2.clean_rfm(rfm)
g = d[d["sensor_id"] == args.sensor].sort_values("server_time").reset_index(drop=True)
seg = v2.split_segments(g, c2)
gs = seg[seg["segment"] == args.segment - 1]
if gs.empty:
    raise SystemExit(f"No segment {args.segment} found for {args.sensor}. "
                      f"Available: {sorted(seg['segment'].unique() + 1)}")

built = v2.build_segment(gs, c2)
if built is None:
    raise SystemExit("Segment too short/sparse to build windows from.")
F, ref = built

if args.list_sessions:
    S_real, sig_real, _ = v3.track(v3.session_table(F), cfg, rebaseline=False)
    view = S_real[["t", "u", "v", "n", "state", "shift_deg", "z"]].copy()
    view["t"] = view["t"].dt.tz_convert(cfg.tz)
    pd.set_option("display.width", 200)
    print(f"{args.sensor}_s{args.segment}: {len(F)} windows, sigma_q={sig_real:.3f} deg\n")
    print(view.round(3).to_string())
    print("\nPick an onset from a BASE/OK/SETTLING-after-arm-only row, safely BEFORE the first EVENT/ALARM/DRIFT row.")
    print("Then re-run with --onset_time '<that IST timestamp, or a bit after it>' (quotes required).")
    raise SystemExit

if args.onset_time is not None:
    ts = pd.Timestamp(args.onset_time)
    ts = ts.tz_localize(cfg.tz) if ts.tzinfo is None else ts
    ts = ts.tz_convert(F.index.tz)
    onset = int(np.searchsorted(F.index.to_numpy(), ts.to_numpy()))
    onset = min(max(onset, 0), len(F) - 1)
else:
    onset = int(len(F) * args.onset_frac)

n_pre_sessions = F["session"].iloc[:onset].nunique() if onset > 0 else 0
print(f"{args.sensor}_s{args.segment}: {len(F)} windows total, injecting at window {onset} "
      f"({F.index[onset]} IST={F.index[onset].tz_convert(cfg.tz)})")
print(f"Sessions before onset: {n_pre_sessions} (level detector needs >= {cfg.arm_sessions} to arm -- "
      f"if this is below that, the level channel will show OK/UNARMED for the whole test regardless of the fault)")

# calibrate on the pre-onset (clean) data only -- this is what "training" the detector on your quiet data means
Fc = F.iloc[:onset]
S = v3.session_table(Fc)
Sc, sig, _ = v3.track(S, cfg, rebaseline=False)
qr = Sc.loc[Sc.state.isin(["OK", "DRIFT"]), "drift_deg_day"].dropna()
dthr = max(cfg.drift_floor, float(np.quantile(qr, cfg.drift_q))) if len(qr) >= 10 else cfg.drift_floor
tex = v2.Model(c2).fit(Fc)

Fx = v3.inject_v3(F, args.kind, args.mag, onset, args.hours)
sc = v3.fuse(Fx, tex, cfg, sig, dthr)

flagged = sc["status"].isin(["ALARM", "DRIFT", "SENSOR", "WARNING"]).to_numpy()
hit = np.flatnonzero(flagged[onset:])

print(f"\nInjected: {args.kind} mag={args.mag}" + (f" over {args.hours}h" if args.kind == "ramp" else ""))
print(f"Sensor baseline noise (sigma_q): {sig:.3f} deg")
print(f"Drift threshold used: {dthr:.3f} deg/day")
if len(hit):
    delay_h = (Fx.index[onset + hit[0]] - Fx.index[onset]).total_seconds() / 3600
    print(f"DETECTED -> status={sc['status'].iloc[onset + hit[0]]} at window {onset+hit[0]}, "
          f"delay = {delay_h:.2f} hours after onset")
else:
    print("NOT DETECTED in the rest of this segment.")

print("\nWindow-by-window status around the onset (10 before, 20 after):")
lo, hi = max(0, onset - 10), min(len(Fx), onset + 20)
view = pd.DataFrame({"time": Fx.index[lo:hi], "dev_u": Fx["dev_u"].to_numpy()[lo:hi],
                     "status": sc["status"].to_numpy()[lo:hi]})
print(view.to_string(index=False))

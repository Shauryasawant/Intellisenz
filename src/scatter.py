import numpy as np, pandas as pd
for name in ["RFM_0001_s14", "RFM_0004_s2", "RFM_0004_s3", "RFM_0003_s3"]:
    S = pd.read_csv(f"runs/{name}_sessions.csv", index_col=0, parse_dates=["t"])
    S["t_ist"] = S.t.dt.tz_convert("Asia/Kolkata").dt.strftime("%m-%d %H:%M")
    q = S[S.n >= 3]
    mu = q[["u", "v"]].median()
    d = np.hypot(q.u - mu.u, q.v - mu.v)
    print(f"\n{name}: {len(q)} qualifying sessions, scatter around median: "
          f"p50 {d.median():.2f}  p90 {d.quantile(.9):.2f}  max {d.max():.2f} deg")
    print(q[["t_ist", "u", "v", "n", "state"]].round(3).to_string())
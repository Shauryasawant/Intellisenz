import glob
import numpy as np
import pandas as pd

for path in sorted(glob.glob("runs/RFM_*_alerts.csv")):
    f = pd.read_csv(path, index_col=0, parse_dates=True)
    f["t"] = f.index
    sm = f.groupby("session").agg(start=("t", "first"), u=("dev_u", "median"),
                                  v=("dev_v", "median"), n=("dev_u", "size"))
    sm["jump_deg"] = np.hypot(sm.u.diff(), sm.v.diff())
    print(f"\n--- {path.split('/')[-1][:8]}: sessions where tilt jumped > 0.5 deg ---")
    print(sm[sm.jump_deg > 0.5].round(2).to_string())
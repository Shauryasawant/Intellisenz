"""Replay the newest raw-data segment through its current realtime alarm."""

import json
from importlib import import_module
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "test"))

fast_refs = import_module("fast_refs")
load_fast_alarms = fast_refs.load_fast_alarms
latest_segment_numbers = fast_refs.latest_segment_numbers
replay = fast_refs.replay

from intellisenz.models.SHM_modelv2 import Cfg, clean_rfm, split_segments
from intellisenz.preprocess.preprocessing import load_raw_export, preprocess_raw_export

CSV_PATH = ROOT / "influx_data.csv"
RUNS_PATH = ROOT / "runs" / "model_v2"
SENSOR_ID = "RFM_0002"

raw = preprocess_raw_export(load_raw_export(str(CSV_PATH)))["RFM"]
d = clean_rfm(raw)
saved_cfg = json.loads((RUNS_PATH / "summary.json").read_text()).get("cfg", {})
cfg = Cfg(**saved_cfg)
latest_seg = latest_segment_numbers(d, cfg)
alarms = load_fast_alarms(str(RUNS_PATH / "epochs.csv"), latest_seg=latest_seg)

if SENSOR_ID not in latest_seg:
    raise ValueError(f"No qualifying raw-data segment found for {SENSOR_ID}")
if SENSOR_ID not in alarms:
    raise KeyError(f"No realtime alarm configured for {SENSOR_ID}")

sensor_rows = d[d["sensor_id"] == SENSOR_ID].reset_index(drop=True)
segmented = split_segments(sensor_rows, cfg)
segment_id = latest_seg[SENSOR_ID] - 1
df = segmented[segmented["segment"] == segment_id].sort_values("server_time")
if df.empty:
    raise ValueError(f"Latest segment {latest_seg[SENSOR_ID]} has no replayable readings")


def process_reading(sensor_id: str, server_time_seconds: float, ax: float, ay: float, az: float):
    """Pass one incoming sensor reading through that sensor's FastAlarm."""
    alarm = alarms.get(sensor_id)
    if alarm is None:
        return None
    return alarm.update(server_time_seconds, (ax, ay, az))


alerts = replay(df, alarms[SENSOR_ID])
print(
    f"Latest raw segment replay for {SENSOR_ID} (segment {latest_seg[SENSOR_ID]}): "
    f"{len(df)} readings, {len(alerts)} alert(s), reference="
    f"{'armed' if alarms[SENSOR_ID].ref is not None else 'unarmed'}"
)
if not alerts.empty:
    print(alerts.groupby("level").size().rename("count").to_string())
    print(alerts.head(20).to_string(index=False))
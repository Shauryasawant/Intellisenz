from pathlib import Path
from importlib import import_module
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "test"))

fast_refs = import_module("fast_refs")
load_fast_alarms = fast_refs.load_fast_alarms
replay = fast_refs.replay
from intellisenz.models.SHM_modelv2 import clean_rfm
from intellisenz.preprocess.preprocessing import load_raw_export, preprocess_raw_export

CSV_PATH = ROOT / "influx_data.csv"
RUNS_PATH = ROOT / "runs"
SENSOR_ID = "RFM_0002"

rfm = preprocess_raw_export(load_raw_export(str(CSV_PATH)))["RFM"]
d = clean_rfm(rfm)
sensor_data = d[d["sensor_id"] == SENSOR_ID].sort_values("server_time")
epochs_path = RUNS_PATH / "epochs.csv"
segments_path = RUNS_PATH / "segments.csv"
alarms = load_fast_alarms(str(epochs_path), str(segments_path))

if SENSOR_ID not in alarms:
	raise KeyError(f"No fast alarm was configured for {SENSOR_ID}")

segment_rows = pd.read_csv(segments_path)
segment_rows["sensor_id"] = segment_rows["segment"].str.rsplit("_s", n=1).str[0]
segment_rows["segment_number"] = segment_rows["segment"].str.rsplit("_s", n=1).str[1].astype(int)
sensor_segments = segment_rows[segment_rows["sensor_id"] == SENSOR_ID]
if sensor_segments.empty:
	raise ValueError(f"No segment metadata found for {SENSOR_ID}")
latest_segment = sensor_segments.sort_values("segment_number").iloc[-1]["segment"]

session_path = RUNS_PATH / f"{latest_segment}_sessions.csv"
session_rows = pd.read_csv(session_path, parse_dates=["t"])
epoch_rows = pd.read_csv(epochs_path)
latest_epoch = epoch_rows[epoch_rows["segment"] == latest_segment]
if latest_epoch.empty:
	replay_start = session_rows["t"].min()
else:
	epoch_number = int(latest_epoch["epoch"].max())
	epoch_sessions = session_rows[
		(session_rows["epoch"] == epoch_number)
		& session_rows["state"].isin(["BASE", "OK"])
	]
	replay_start = epoch_sessions["t"].min() if not epoch_sessions.empty else session_rows["t"].min()

replay_start = pd.to_datetime(replay_start, utc=True)
replay_end = pd.to_datetime(session_rows["t"].max(), utc=True)
df = sensor_data[
	(sensor_data["server_time"] >= replay_start)
	& (sensor_data["server_time"] <= replay_end)
]
unmodeled_readings = int((sensor_data["server_time"] > replay_end).sum())


def process_reading(sensor_id: str, server_time_seconds: float, ax: float, ay: float, az: float):
	"""Pass one incoming sensor reading through that sensor's FastAlarm."""
	alarm = alarms.get(sensor_id)
	if alarm is None:
		return None
	return alarm.update(server_time_seconds, (ax, ay, az))


historical_alarms = load_fast_alarms(str(RUNS_PATH / "epochs.csv"), str(RUNS_PATH / "segments.csv"))
alerts = replay(df, historical_alarms[SENSOR_ID])
print(f"Historical replay for {SENSOR_ID} from {replay_start} through {replay_end}: {len(alerts)} alert(s)")
if unmodeled_readings:
	print(f"Excluded {unmodeled_readings} newer reading(s): regenerate model outputs before evaluating them.")
if not alerts.empty:
	print(alerts.groupby("level").size().rename("count").to_string())
	print(alerts.head(20).to_string(index=False))
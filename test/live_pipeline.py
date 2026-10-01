"""
live_pipeline.py -- one refresh cycle: InfluxDB -> CSV -> SHM_modelv4 -> dashboard JSON.

Reuses your existing code unchanged:
  * data.py                         (Influx settings loader)
  * intellisenz.preprocess          (load_raw_export / preprocess_raw_export)
  * intellisenz.models.SHM_modelv4  (run_all / export_dashboard_json)

Run from the project root (the folder that contains the `intellisenz` package).
"""
import csv
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from influxdb_client import InfluxDBClient

from data import load_influx_settings  # your existing helper

logger = logging.getLogger("live_pipeline")

WORK_DIR = Path(os.getenv("SHM_WORK_DIR", "runs_live"))
CSV_PATH = WORK_DIR / "influx_data.csv"
POSITIONS_FILE = Path(os.getenv("SHM_POSITIONS", "positions.json"))   # optional {"sensor_id": [x, y, z]}
# The model needs history (arming, weekly drift), so every cycle re-reads this whole window.
INFLUX_RANGE = os.getenv("INFLUX_RANGE", "-30d")


def fetch_to_csv(csv_path: Path = CSV_PATH, rng: str = INFLUX_RANGE) -> int:
    """Same query/CSV layout as data.py, but with a configurable range and output path."""
    s = load_influx_settings()
    client = InfluxDBClient(url=s.url, token=s.token, org=s.org, verify_ssl=s.verify_ssl, timeout=120_000)
    try:
        flux = f'''
            from(bucket: "{s.bucket}")
              |> range(start: {rng})
              |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
        '''
        rows = []
        for table in client.query_api().query(query=flux, org=s.org):
            for rec in table.records:
                row = {"time": rec.get_time(), "measurement": rec.get_measurement()}
                for k, v in rec.values.items():
                    if k.startswith("_") or k in ("result", "table"):
                        continue
                    row[k] = v
                rows.append(row)
    finally:
        client.close()

    if not rows:
        raise RuntimeError(f"InfluxDB returned no rows for range {rng}")

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    cols = sorted({k for r in rows for k in r})
    tmp = csv_path.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(csv_path)                      # atomic swap
    logger.info("Fetched %d rows from InfluxDB", len(rows))
    return len(rows)


def run_model(csv_path: Path = CSV_PATH, out_dir: Path = WORK_DIR / "out") -> dict:
    """Exactly what SHM_modelv4.main() does, minus argparse/printing."""
    from intellisenz.models import SHM_modelv2 as v2
    from intellisenz.models import SHM_modelv4 as v4
    from intellisenz.preprocess.preprocessing import load_raw_export, preprocess_raw_export

    frames = preprocess_raw_export(load_raw_export(str(csv_path)))
    rfm = v2.clean_rfm(frames["RFM"])
    res = v4.run_all(frames["RFL"], v4.Cfg4(), str(out_dir), rfm_df=rfm)
    json_path = v4.export_dashboard_json(res, str(out_dir))
    return json.loads(Path(json_path).read_text())


def refresh() -> dict:
    """Full cycle. Returns the payload the dashboard consumes."""
    n_rows = fetch_to_csv()
    payload = run_model()
    payload["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    payload["rows_fetched"] = n_rows
    payload["influx_range"] = INFLUX_RANGE
    if POSITIONS_FILE.exists():
        payload["positions"] = json.loads(POSITIONS_FILE.read_text())
    return payload





# Select-String -Path live_server.py -Pattern "^import|^from"
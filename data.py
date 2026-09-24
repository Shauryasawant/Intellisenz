"""
data.py — Manual one-shot data fetch & CSV export utility.

Fetches the last 2 minutes of sensor data from InfluxDB
and saves it as CSV.

Usage:
    python data.py

Expected environment variables:
    INFLUX_URL
    INFLUX_TOKEN
    INFLUX_ORG
    INFLUX_BUCKET
    INFLUX_VERIFY_SSL
"""

import csv
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from influxdb_client import InfluxDBClient
from dotenv import load_dotenv


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(message)s"
)

logger = logging.getLogger(__name__)

OUTPUT_FILE = "influx_data.csv"


@dataclass(frozen=True)
class InfluxSettings:
    url: str
    token: str
    org: str
    bucket: str
    verify_ssl: bool


def load_influx_settings() -> InfluxSettings:
    # Load .env from the same directory as this script
    env_file = Path(__file__).resolve().parent / ".env"

    if env_file.exists():
        load_dotenv(env_file)

    settings = {
        "INFLUX_URL": os.getenv("INFLUX_URL"),
        "INFLUX_TOKEN": os.getenv("INFLUX_TOKEN"),
        "INFLUX_ORG": os.getenv("INFLUX_ORG"),
        "INFLUX_BUCKET": os.getenv("INFLUX_BUCKET"),
        "INFLUX_VERIFY_SSL": os.getenv(
            "INFLUX_VERIFY_SSL",
            "false"
        ),
    }

    missing = [
        name
        for name, value in settings.items()
        if name != "INFLUX_VERIFY_SSL" and not value
    ]

    if missing:
        raise RuntimeError(
            "Missing InfluxDB environment variables: "
            + ", ".join(missing)
        )

    return InfluxSettings(
        url=settings["INFLUX_URL"],
        token=settings["INFLUX_TOKEN"],
        org=settings["INFLUX_ORG"],
        bucket=settings["INFLUX_BUCKET"],
        verify_ssl=settings["INFLUX_VERIFY_SSL"].lower()
        in {"1", "true", "yes", "on"},
    )


def main() -> None:
    settings = load_influx_settings()

    client = InfluxDBClient(
        url=settings.url,
        token=settings.token,
        org=settings.org,
        verify_ssl=settings.verify_ssl,
    )

    try:
        query_api = client.query_api()

        flux_query = f"""
            from(bucket: "{settings.bucket}")
                |> range(start: -1d)
                |> pivot(
                    rowKey: ["_time"],
                    columnKey: ["_field"],
                    valueColumn: "_value"
                )
        """

        tables = query_api.query(
            query=flux_query,
            org=settings.org
        )

        rows = []

        for table in tables:
            for record in table.records:
                row = {
                    "time": record.get_time(),
                    "measurement": record.get_measurement(),
                }

                for key, value in record.values.items():
                    if key.startswith("_") or key in ("result", "table"):
                        continue

                    row[key] = value

                rows.append(row)

        logger.info(f"Fetched {len(rows)} record(s).")

        if rows:
            columns = sorted(
                {key for row in rows for key in row}
            )

            with open(
                OUTPUT_FILE,
                "w",
                newline="",
                encoding="utf-8",
            ) as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=columns
                )

                writer.writeheader()
                writer.writerows(rows)

            logger.info(
                f"CSV written to {OUTPUT_FILE}"
            )
        else:
            logger.info("No data found in the last 2 minutes.")

    finally:
        client.close()


if __name__ == "__main__":
    main()


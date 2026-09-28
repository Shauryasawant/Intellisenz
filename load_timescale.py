"""
load_timescale.py

Reads the raw sensor export (influx_data.csv), runs it through the preprocessing
module, and stores the result in TimescaleDB.

Database:
    PostgreSQL / TimescaleDB

Usage:
    python load_timescale.py
"""

import logging

import psycopg2
from psycopg2 import extras

from intellisenz.preprocess.preprocessing import (
    frame_to_records,
    load_raw_export,
    preprocess_raw_export,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(message)s"
)

logger = logging.getLogger(__name__)

CSV_FILE = "influx_data.csv"

DB_CONFIG = {
    "host": "localhost",
    "port": 5433,
    "database": "intellisenz",
    "user": "postgres",
    "password": "postgres",
}


def create_table(connection) -> None:
    """
    Create the sensor_data table if it does not already exist.
    """

    with connection.cursor() as cursor:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS sensor_data (
                id BIGSERIAL PRIMARY KEY,
                time TIMESTAMPTZ NOT NULL,
                measurement TEXT,
                data JSONB
            );
            """
        )

    connection.commit()

    logger.info("Table sensor_data is ready.")


def build_records(csv_file: str = CSV_FILE) -> list:
    """
    Read the CSV and preprocess it into (time, measurement, data) tuples.
    """

    df_raw = load_raw_export(csv_file)
    frames = preprocess_raw_export(df_raw)

    records = []
    for measurement, frame in frames.items():
        if frame.empty:
            logger.info("No %s rows found; skipping this device type.", measurement)
            continue
        records.extend(frame_to_records(frame))

    records.sort(key=lambda r: r[0])
    return records


def load_records(connection, records: list) -> None:
    """
    Replace the existing sensor_data contents with the latest records.
    Runs in a single transaction, so a failure leaves the old data intact.
    """

    try:
        with connection.cursor() as cursor:

            # Remove previous batch
            #cursor.execute("TRUNCATE TABLE sensor_data;")

            extras.execute_values(
                cursor,
                "INSERT INTO sensor_data (time, measurement, data) VALUES %s",
                [(ts, measurement, extras.Json(data)) for ts, measurement, data in records],
                page_size=1000,
            )

        connection.commit()

    except Exception:
        connection.rollback()
        raise

    logger.info(
        f"Inserted {len(records)} record(s) into TimescaleDB."
    )


def main() -> None:

    records = build_records()
    if not records:
        raise RuntimeError("No valid sensor records found; database was not modified.")

    connection = psycopg2.connect(**DB_CONFIG)

    try:
        create_table(connection)
        load_records(connection, records)

    finally:
        connection.close()

    logger.info("TimescaleDB loading completed.")


if __name__ == "__main__":
    main()
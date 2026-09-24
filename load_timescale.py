"""
load_timescale.py

Reads sensor data from influx_data.csv and stores it in TimescaleDB.

Database:
    PostgreSQL / TimescaleDB

Usage:
    python load_timescale.py
"""

import csv
import logging
from datetime import datetime

import psycopg2
from psycopg2 import extras

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


def load_csv(connection) -> None:
    """
    Replace the existing sensor_data contents
    with the latest CSV data.
    """

    with open(
        CSV_FILE,
        "r",
        newline="",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        rows_inserted = 0

        with connection.cursor() as cursor:

            # Remove previous batch
            cursor.execute("TRUNCATE TABLE sensor_data;")

            for row in reader:

                timestamp = row.pop("time")
                measurement = row.pop("measurement", None)

                timestamp = datetime.fromisoformat(
                    timestamp.replace("Z", "+00:00")
                )

                cursor.execute(
                    """
                    INSERT INTO sensor_data (
                        time,
                        measurement,
                        data
                    )
                    VALUES (%s, %s, %s);
                    """,
                    (
                        timestamp,
                        measurement,
                        extras.Json(row),
                    )
                )

                rows_inserted += 1

        connection.commit()

    logger.info(
        f"Inserted {rows_inserted} record(s) into TimescaleDB."
    )


def main() -> None:

    connection = psycopg2.connect(**DB_CONFIG)

    try:
        create_table(connection)
        load_csv(connection)

    finally:
        connection.close()

    logger.info("TimescaleDB loading completed.")


if __name__ == "__main__":
    main()

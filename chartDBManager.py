"""Persist the generated sector-performance chart PNG into a Neon/Postgres
database table. Only one row is ever kept -- each save wipes the table and
inserts the freshly generated PNG as a BYTEA object.
"""
from __future__ import annotations

import os
from datetime import datetime

import psycopg2


class ChartDBManager:
    TABLE = "sectorchart"

    def __init__(self):
        self.conn_string = os.getenv("DATABASE_URL")
        if not self.conn_string:
            raise RuntimeError("Missing DATABASE_URL. Set it in your .env file.")

    def _ensure_table(self, cur) -> None:
        """Create the table if it doesn't exist yet."""
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.TABLE} (
                id SERIAL PRIMARY KEY,
                "chartName" TEXT NOT NULL,
                "chartImage" BYTEA NOT NULL,
                "createdAt" TIMESTAMP NOT NULL DEFAULT NOW()
            );
            """
        )

    def save_chart(self, png_path: str) -> None:
        """Replace the single stored chart row with the PNG at png_path.

        Deletes all existing rows first, then inserts the new image, so
        exactly one row exists in the table after a successful call.
        """
        if not os.path.isfile(png_path):
            raise FileNotFoundError(png_path)

        with open(png_path, "rb") as f:
            png_bytes = f.read()

        try:
            with psycopg2.connect(self.conn_string) as conn:
                with conn.cursor() as cur:
                    self._ensure_table(cur)
                    cur.execute(f'DELETE FROM {self.TABLE};')
                    cur.execute(
                        f'INSERT INTO {self.TABLE} ("chartName", "chartImage", "createdAt") '
                        f'VALUES (%s, %s, %s);',
                        (os.path.basename(png_path), psycopg2.Binary(png_bytes), datetime.now()),
                    )
        except psycopg2.Error as e:
            print(f"Error connecting to or querying the database: {e}")
        return

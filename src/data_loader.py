import sqlite3
from pathlib import Path

import pandas as pd


class SQLiteDeliveryLoader:
    def __init__(self, db_path, delivery_table="deliveries", feedback_table="feedback"):
        self.db_path = Path(db_path)
        self.delivery_table = delivery_table
        self.feedback_table = feedback_table

    def load(self):
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")

        with sqlite3.connect(self.db_path) as connection:
            deliveries = pd.read_sql(f"SELECT * FROM {self.delivery_table}", connection)
            feedback = pd.read_sql(f"SELECT * FROM {self.feedback_table}", connection)

        return deliveries, feedback

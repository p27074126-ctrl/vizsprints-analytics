import pandas as pd
from sqlalchemy import create_engine, Table, Column, MetaData, Integer, String, Text, DateTime
from sqlalchemy.dialects.sqlite import JSON
from datetime import datetime
import sqlite3

DB_FILE = "vizsprints.db"

def create_tables(engine):
    meta = MetaData()
    users = Table(
        "users", meta,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("user_id", String, unique=True, nullable=False),
        Column("joined_at", String),
        Column("device", String),
        Column("country", String),
        Column("subscription_status", String),
        Column("ab_group", String)
    )
    events = Table(
        "events", meta,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("event_id", String, unique=True, nullable=False),
        Column("user_id", String, nullable=False),
        Column("event_name", String),
        Column("timestamp", String),
        Column("metadata", Text)
    )
    meta.create_all(engine)
    return users, events

def load_csvs(engine):
    users_df = pd.read_csv("users.csv", parse_dates=False)
    events_df = pd.read_csv("events.csv", parse_dates=False)
    # Basic referential integrity check
    user_ids = set(users_df['user_id'].astype(str).tolist())
    events_df = events_df[events_df['user_id'].isin(user_ids)].copy()
    print(f"{len(events_df)} events remain after filtering unknown user_ids.")

    users_df.to_sql('users', engine, if_exists='replace', index=False)
    events_df.to_sql('events', engine, if_exists='replace', index=False)
    print("Loaded users and events into SQLite.")

if __name__ == "__main__":
    engine = create_engine(f"sqlite:///{DB_FILE}", connect_args={"check_same_thread": False})
    create_tables(engine)
    load_csvs(engine)
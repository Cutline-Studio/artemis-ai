"""Database access for the business engine: SQLite for development and tests, PostgreSQL in production.

ARTEMIS_DATABASE_URL=postgresql://user:pass@host:5432/artemis?sslmode=require  -> PostgreSQL
anything else (a file path, or ":memory:")                                      -> SQLite
SQL is written with "?" placeholders; they are translated for PostgreSQL.
"""
from __future__ import annotations

import sqlite3
import threading

SCHEMA = """
CREATE TABLE IF NOT EXISTS accounts (id TEXT PRIMARY KEY, plan TEXT NOT NULL, created {real});
CREATE TABLE IF NOT EXISTS api_keys (hash TEXT PRIMARY KEY, account TEXT NOT NULL, created {real}, revoked INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS usage (account TEXT, day TEXT, kind TEXT, amount {real}, model TEXT, cost_usd {real}, ts {real});
CREATE INDEX IF NOT EXISTS usage_by_day ON usage(account, day, kind);
CREATE INDEX IF NOT EXISTS usage_by_ts ON usage(account, kind, ts);
CREATE TABLE IF NOT EXISTS waitlist (email TEXT PRIMARY KEY, created {real}, source TEXT);
CREATE TABLE IF NOT EXISTS memories (id TEXT PRIMARY KEY, account TEXT NOT NULL, text TEXT NOT NULL, created {real}, source TEXT);
CREATE INDEX IF NOT EXISTS memories_by_account ON memories(account, created);
"""


class Store:
    def __init__(self, url: str):
        self.url = url
        self.postgres = url.startswith(("postgres://", "postgresql://"))
        self.lock = threading.Lock()
        self._connect()
        with self.lock:
            for stmt in filter(None, (s.strip() for s in SCHEMA.format(real="DOUBLE PRECISION" if self.postgres else "REAL").split(";"))):
                self._raw(stmt, ())

    def _connect(self):
        if self.postgres:
            import psycopg
            self.conn = psycopg.connect(self.url, autocommit=True, connect_timeout=10)
        else:
            self.conn = sqlite3.connect(self.url, check_same_thread=False, isolation_level=None)

    def _raw(self, sql: str, params: tuple):
        if self.postgres:
            sql = sql.replace("?", "%s")
        try:
            return self.conn.execute(sql, params)
        except Exception as e:
            if not self.postgres or not self._is_disconnect(e):
                raise
            self._connect()  # the server closed an idle connection; retry once
            return self.conn.execute(sql, params)

    @staticmethod
    def _is_disconnect(e: Exception) -> bool:
        import psycopg
        return isinstance(e, (psycopg.OperationalError, psycopg.InterfaceError))

    def execute(self, sql: str, params: tuple = ()) -> None:
        with self.lock:
            self._raw(sql, params)

    def one(self, sql: str, params: tuple = ()):
        with self.lock:
            return self._raw(sql, params).fetchone()

    def all(self, sql: str, params: tuple = ()) -> list:
        with self.lock:
            return self._raw(sql, params).fetchall()

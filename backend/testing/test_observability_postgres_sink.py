import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from observability import PostgresTraceSink  # noqa: E402


class FakeCursor:
    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, query, params):
        normalized = " ".join(str(query).split())
        self.conn.calls.append(
            {
                "query": normalized,
                "params": params,
            }
        )


class FakeConn:
    def __init__(self):
        self.autocommit = False
        self.closed = 0
        self.calls: list[dict] = []

    def cursor(self):
        return FakeCursor(self)

    def close(self):
        self.closed = 1


class PostgresTraceSinkTests(unittest.TestCase):
    def test_postgres_trace_sink_writes_run_step_and_event_queries(self):
        fake_conn = FakeConn()
        with mock.patch("observability.sinks.psycopg2.connect", return_value=fake_conn):
            sink = PostgresTraceSink("postgresql://example")

            sink.emit(
                {
                    "event_id": "evt-run-start",
                    "event_type": "run.started",
                    "run_id": "run-1",
                    "step_id": None,
                    "step_key": None,
                    "workflow_name": "demo.workflow",
                    "workflow_version": "v1",
                    "seq": 1,
                    "ts": "2026-05-02T00:00:00+00:00",
                    "payload": {
                        "metadata": {"mode": "shadow"},
                        "tags": {"team": "research"},
                        "correlation_ids": {"request_id": "abc"},
                    },
                }
            )
            sink.emit(
                {
                    "event_id": "evt-step-start",
                    "event_type": "step.started",
                    "run_id": "run-1",
                    "step_id": "step-1",
                    "step_key": "build_context",
                    "workflow_name": "demo.workflow",
                    "workflow_version": "v1",
                    "seq": 2,
                    "ts": "2026-05-02T00:00:01+00:00",
                    "payload": {
                        "step_key": "build_context",
                        "kind": "prepare",
                        "attempt": 1,
                        "metadata": {"scope": "live"},
                        "parent_step_id": None,
                    },
                }
            )
            sink.emit(
                {
                    "event_id": "evt-step-complete",
                    "event_type": "step.completed",
                    "run_id": "run-1",
                    "step_id": "step-1",
                    "step_key": "build_context",
                    "workflow_name": "demo.workflow",
                    "workflow_version": "v1",
                    "seq": 3,
                    "ts": "2026-05-02T00:00:02+00:00",
                    "payload": {
                        "status": "succeeded",
                        "summary": {"rows": 20},
                        "started_at": "2026-05-02T00:00:01+00:00",
                        "ended_at": "2026-05-02T00:00:02+00:00",
                        "duration_ms": 1000,
                        "kind": "prepare",
                        "attempt": 1,
                    },
                }
            )
            sink.emit(
                {
                    "event_id": "evt-run-complete",
                    "event_type": "run.completed",
                    "run_id": "run-1",
                    "step_id": None,
                    "step_key": None,
                    "workflow_name": "demo.workflow",
                    "workflow_version": "v1",
                    "seq": 4,
                    "ts": "2026-05-02T00:00:03+00:00",
                    "payload": {
                        "status": "succeeded",
                        "outcome_code": "NO_CHANGE",
                        "summary": {"verdict": "hold"},
                        "started_at": "2026-05-02T00:00:00+00:00",
                        "ended_at": "2026-05-02T00:00:03+00:00",
                        "duration_ms": 3000,
                    },
                }
            )

            self.assertTrue(fake_conn.autocommit)
            self.assertEqual(len(fake_conn.calls), 8)
            self.assertIn("INSERT INTO trace_runs", fake_conn.calls[0]["query"])
            self.assertIn("INSERT INTO trace_events", fake_conn.calls[1]["query"])
            self.assertIn("INSERT INTO trace_steps", fake_conn.calls[2]["query"])
            self.assertIn("INSERT INTO trace_events", fake_conn.calls[3]["query"])
            self.assertIn("INSERT INTO trace_steps", fake_conn.calls[4]["query"])
            self.assertIn("INSERT INTO trace_runs", fake_conn.calls[6]["query"])
            self.assertEqual(fake_conn.calls[1]["params"][6], "run.started")
            self.assertEqual(fake_conn.calls[7]["params"][6], "run.completed")

            sink.close()
            self.assertEqual(fake_conn.closed, 1)


if __name__ == "__main__":
    unittest.main()

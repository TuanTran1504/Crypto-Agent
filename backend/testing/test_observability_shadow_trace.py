import json
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT
SCHEDULE_DIR = ROOT / "schedule"
TESTING_DIR = ROOT / "testing"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(SCHEDULE_DIR) not in sys.path:
    sys.path.insert(0, str(SCHEDULE_DIR))
if str(TESTING_DIR) not in sys.path:
    sys.path.insert(0, str(TESTING_DIR))

from observability import (  # noqa: E402
    JsonlTraceSink,
    LocalBlobStore,
    RedactionRule,
    TraceRedactor,
    Tracer,
)
import run_policy_review_shadow_multi_agent as shadow  # noqa: E402


@dataclass
class FakeGuard:
    decision: str
    reason: str
    guard_config: dict
    context: dict


class FakeResponse:
    def __init__(self, content: str):
        self.choices = [
            SimpleNamespace(
                message=SimpleNamespace(content=content),
            )
        ]


class FakeCompletions:
    def __init__(self, payloads: list[str]):
        self._payloads = list(payloads)
        self.requests: list[dict] = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return FakeResponse(self._payloads.pop(0))


class FakeClient:
    def __init__(self, payloads: list[str]):
        self.chat = SimpleNamespace(completions=FakeCompletions(payloads))


def _read_events(trace_path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class ObservabilityTraceTests(unittest.TestCase):
    def test_tracer_redacts_and_blob_offloads_large_artifact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_path = Path(tmpdir) / "trace.jsonl"
            blob_dir = Path(tmpdir) / "blobs"
            tracer = Tracer(
                sinks=[JsonlTraceSink(trace_path)],
                redactor=TraceRedactor(
                    [
                        RedactionRule("payload.secret", "mask"),
                    ]
                ),
                blob_store=LocalBlobStore(blob_dir),
                blob_threshold_bytes=200,
            )

            with tracer.start_run("demo.workflow", workflow_version="test") as run:
                with run.step("large_payload", kind="prepare") as step:
                    step.record_artifact(
                        "payload",
                        {
                            "secret": "hide-me",
                            "notes": "x" * 600,
                        },
                        role="input",
                        redaction_path=("payload",),
                    )

            tracer.close()

            events = _read_events(trace_path)
            artifact_events = [event for event in events if event["event_type"] == "artifact.captured"]
            self.assertEqual(len(artifact_events), 1)
            artifact_payload = artifact_events[0]["payload"]
            self.assertEqual(artifact_payload["storage"], "blob")
            blob_path = Path(artifact_payload["blob"]["blob_uri"])
            self.assertTrue(blob_path.exists())
            blob_text = blob_path.read_text(encoding="utf-8")
            self.assertNotIn("hide-me", blob_text)
            self.assertIn("[REDACTED]", blob_text)

    def test_shadow_preview_emits_trace_lifecycle_and_redacted_llm_payloads(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_path = Path(tmpdir) / "shadow_trace.jsonl"
            fake_client = FakeClient(
                [
                    json.dumps(
                        {
                            "summary": "Range trades weakened recently.",
                            "key_findings": ["Setup D underperformed."],
                            "candidate_hypotheses": ["Disable Setup D."],
                            "confidence": 0.72,
                        }
                    ),
                    json.dumps(
                        {
                            "decision": "PROPOSE_CHANGE",
                            "reason": "Setup D has clustered losses.",
                            "confidence": 0.64,
                            "changes": [
                                {
                                    "path": "setups.D.enabled",
                                    "value": False,
                                    "because": "Recent range trades lost money.",
                                }
                            ],
                        }
                    ),
                    json.dumps(
                        {
                            "decision": "APPROVE",
                            "reason": "Minimal, conservative patch.",
                            "confidence": 0.67,
                            "approved_changes": [
                                {
                                    "path": "setups.D.enabled",
                                    "value": False,
                                    "because": "Recent range trades lost money.",
                                }
                            ],
                        }
                    ),
                ]
            )

            review_input = {
                "engine_name": "llm_live",
                "account_type": "live",
                "as_of": "2026-05-02T00:00:00+00:00",
                "guard": {
                    "decision": "ALLOW_REVIEW",
                    "reason": "Enough evidence",
                },
                "active_policy": {
                    "id": 101,
                    "version": 7,
                },
                "deterministic_checks": {
                    "qa_verdict": "OK",
                    "allow_llm_proposal": True,
                    "clear_signal_for_risk_increase": False,
                    "clear_signal_evidence": {},
                    "issues": [],
                    "warnings": [],
                    "risk_constraints": {
                        "allow_risk_increase": False,
                        "max_changed_keys": 2,
                    },
                },
                "evidence_pack": {
                    "windows": {
                        "last_50_trades": {
                            "closed_trades": 50,
                            "total_pnl_usdt": -9.3,
                        }
                    }
                },
            }
            active_row = {
                "id": 101,
                "version": 7,
                "policy_json": {
                    "global": {
                        "trade_min_rr": 1.5,
                        "trade_min_rr_range": 1.2,
                    },
                    "symbols": {
                        "BTC": {
                            "enabled": True,
                            "position_risk_pct": 0.01,
                        }
                    },
                    "setups": {
                        "D": {"enabled": True},
                    },
                },
            }
            fake_guard = FakeGuard(
                decision="ALLOW_REVIEW",
                reason="Enough evidence",
                guard_config={},
                context={},
            )

            with mock.patch.dict("os.environ", {"DATABASE_URL": "postgresql://shadow-test"}):
                with mock.patch.object(
                    shadow.prod,
                    "build_policy_review_context",
                    return_value=(
                        review_input,
                        active_row,
                        fake_guard,
                        shadow.ReviewGuardConfig(trades_table="trades_live"),
                    ),
                ):
                    with mock.patch.object(
                        shadow.prod,
                        "_get_llm_client_and_model",
                        return_value=(fake_client, "openai", "gpt-test"),
                    ):
                        result = shadow.run_multi_agent_shadow_preview(
                            verbose_steps=False,
                            trace_output_path=str(trace_path),
                        )

            self.assertEqual(result["final_decision"], "PROPOSE_CHANGE")
            self.assertEqual(result["trace"]["output_path"], str(trace_path))

            events = _read_events(trace_path)
            raw_trace = trace_path.read_text(encoding="utf-8")
            self.assertIn("run.started", raw_trace)
            self.assertIn("run.completed", raw_trace)
            self.assertNotIn("Return JSON only.", raw_trace)

            completed_step_keys = {
                event["step_key"]
                for event in events
                if event["event_type"] == "step.completed"
            }
            self.assertTrue(
                {
                    "build_review_context",
                    "deterministic_gate_summary",
                    "agent_analyst",
                    "agent_proposer",
                    "agent_critic",
                    "candidate_patch_selection",
                    "patch_validation",
                    "final_shadow_decision",
                }.issubset(completed_step_keys)
            )

            run_completed = [event for event in events if event["event_type"] == "run.completed"]
            self.assertEqual(len(run_completed), 1)
            self.assertEqual(run_completed[0]["payload"]["outcome_code"], "PROPOSE_CHANGE")

            raw_response_events = [
                event
                for event in events
                if event["event_type"] == "artifact.captured"
                and event["payload"]["name"] == "response.raw_text"
            ]
            self.assertEqual(len(raw_response_events), 3)
            for event in raw_response_events:
                self.assertEqual(event["payload"]["storage"], "inline")
                self.assertEqual(event["payload"]["value"]["mode"], "summary_only")


if __name__ == "__main__":
    unittest.main()

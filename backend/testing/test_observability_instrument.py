import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from observability import MemoryTraceSink, Tracer, trace_llm_call, trace_tool_call  # noqa: E402


class ObservabilityInstrumentTests(unittest.TestCase):
    def test_trace_llm_call_records_request_response_and_decision(self):
        sink = MemoryTraceSink()
        tracer = Tracer(sinks=[sink])

        with tracer.start_run("demo.instrument", workflow_version="test") as run:
            parsed = trace_llm_call(
                run,
                step_key="agent_researcher",
                agent_name="researcher",
                provider="openai",
                model="gpt-test",
                messages=[
                    {"role": "system", "content": "System guidance"},
                    {"role": "user", "content": "Analyze BTC"},
                ],
                prompt_input={"symbol": "BTC"},
                invoke=lambda: json.dumps(
                    {
                        "decision": "APPROVE",
                        "reason": "Signal is strong enough.",
                        "score": 0.78,
                    }
                ),
            )

        self.assertEqual(parsed["_meta"]["agent"], "researcher")
        self.assertEqual(parsed["decision"], "APPROVE")

        request_events = [
            event
            for event in sink.events
            if event["event_type"] == "artifact.captured"
            and event["step_key"] == "agent_researcher"
            and event["payload"]["name"] == "request"
        ]
        self.assertEqual(len(request_events), 1)

        decision_events = [
            event
            for event in sink.events
            if event["event_type"] == "decision.recorded"
            and event["step_key"] == "agent_researcher"
        ]
        self.assertEqual(len(decision_events), 1)
        self.assertEqual(decision_events[0]["payload"]["choice"], "APPROVE")

    def test_trace_tool_call_records_result_and_optional_decision(self):
        sink = MemoryTraceSink()
        tracer = Tracer(sinks=[sink])

        with tracer.start_run("demo.instrument", workflow_version="test") as run:
            result = trace_tool_call(
                run,
                step_key="tool_fetch_prices",
                tool_name="fetch_prices",
                request={"symbol": "BTC"},
                invoke=lambda: {"status": "ok", "rows": 2},
                decision_extractor=lambda payload: {
                    "name": "tool_status",
                    "choice": str(payload["status"]).upper(),
                    "metadata": {"rows": payload["rows"]},
                },
            )

        self.assertEqual(result["rows"], 2)

        result_events = [
            event
            for event in sink.events
            if event["event_type"] == "artifact.captured"
            and event["step_key"] == "tool_fetch_prices"
            and event["payload"]["name"] == "result"
        ]
        self.assertEqual(len(result_events), 1)

        decision_events = [
            event
            for event in sink.events
            if event["event_type"] == "decision.recorded"
            and event["step_key"] == "tool_fetch_prices"
        ]
        self.assertEqual(len(decision_events), 1)
        self.assertEqual(decision_events[0]["payload"]["name"], "tool_status")
        self.assertEqual(decision_events[0]["payload"]["choice"], "OK")


if __name__ == "__main__":
    unittest.main()

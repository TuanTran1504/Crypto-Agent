import sys
import unittest
from dataclasses import asdict
from datetime import datetime, timezone
import os
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parent.parent
SCHEDULE_DIR = ROOT / "schedule"
if str(SCHEDULE_DIR) not in sys.path:
    sys.path.insert(0, str(SCHEDULE_DIR))

import run_policy_review as prod  # noqa: E402
from policy_review_guard import ReviewGuardConfig, ReviewGuardResult  # noqa: E402


UTC = timezone.utc


class PolicyReviewRuntimeTests(unittest.TestCase):
    def test_fetch_closed_trade_rows_supports_ascending_order_for_walkforward(self):
        captured: dict[str, object] = {}

        class FakeCursor:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def execute(self, query, params):
                captured["query"] = query
                captured["params"] = params

            def fetchall(self):
                return []

        class FakeConn:
            def cursor(self, cursor_factory=None):
                return FakeCursor()

        with mock.patch.object(prod, "_has_column", return_value=False):
            rows = prod._fetch_closed_trade_rows(
                FakeConn(),
                "trades_live",
                "live",
                since_dt=datetime(2026, 5, 1, tzinfo=UTC),
                until_dt=datetime(2026, 5, 8, tzinfo=UTC),
                limit_rows=3,
                descending=False,
            )

        self.assertEqual(rows, [])
        self.assertIn("ORDER BY closed_at ASC", str(captured.get("query")))

    def test_extract_policy_patch_from_structured_changes(self):
        payload = {
            "changes": [
                {"path": "setups.D.enabled", "value": False},
                {"path": "global.trade_min_rr", "value": 1.8},
            ]
        }

        patch, debug = prod.extract_policy_patch_from_response(payload)

        self.assertEqual(
            patch,
            {
                "setups": {"D": {"enabled": False}},
                "global": {"trade_min_rr": 1.8},
            },
        )
        self.assertEqual(debug["source"], "changes")

    def test_extract_policy_patch_falls_back_to_patch_when_changes_is_none(self):
        payload = {
            "changes": None,
            "patch": {
                "global.trade_min_rr": 1.6,
            },
        }

        patch, debug = prod.extract_policy_patch_from_response(payload)

        self.assertEqual(
            patch,
            {
                "global": {"trade_min_rr": 1.6},
            },
        )
        self.assertEqual(debug["source"], "patch")

    def test_score_policy_patch_on_trades_removes_losing_disabled_setup(self):
        active_policy = {
            "global": {"trade_min_rr": 1.0, "trade_min_rr_range": 1.0},
            "symbols": {"BTC": {"enabled": True}, "ETH": {"enabled": True}},
            "setups": {"A": {"enabled": True}, "D": {"enabled": True}},
        }
        patch = {"setups": {"D": {"enabled": False}}}
        forward_trades = [
            {
                "id": 1,
                "symbol": "BTC",
                "side": "BUY",
                "setup": "Setup D",
                "pnl_usdt": -3.0,
                "entry_price": 100.0,
                "stop_loss": 98.0,
                "take_profit": 104.0,
                "take_profit_2": 0.0,
            },
            {
                "id": 2,
                "symbol": "BTC",
                "side": "BUY",
                "setup": "Setup A",
                "pnl_usdt": 5.0,
                "entry_price": 100.0,
                "stop_loss": 98.0,
                "take_profit": 106.0,
                "take_profit_2": 0.0,
            },
        ]

        score = prod.score_policy_patch_on_trades(active_policy, patch, forward_trades)

        self.assertEqual(score["baseline"]["closed_trades"], 2)
        self.assertEqual(score["candidate"]["closed_trades"], 1)
        self.assertEqual(score["blocked_reason_counts"]["setup_disabled:D"], 1)
        self.assertEqual(score["deltas"]["total_pnl_usdt"], 3.0)

    def test_score_policy_review_candidate_rejects_risk_increase_without_clear_signal(self):
        review_input = {
            "deterministic_checks": {
                "allow_llm_proposal": True,
                "clear_signal_for_risk_increase": False,
                "risk_constraints": {"max_changed_keys": 2},
            }
        }
        sanitized_patch = {"symbols": {"BTC": {"position_risk_pct": 0.02}}}

        score = prod.score_policy_review_candidate(
            review_input,
            sanitized_patch,
            validation_errors=[],
            risk_increase_changes=["symbols.BTC.position_risk_pct: 0.01 -> 0.02"],
        )

        self.assertEqual(score["verdict"], "REJECT")
        self.assertIn("risk increase", " ".join(score["reasons"]))

    def test_engine_behavior_contract_reflects_rule_based_selector(self):
        contract = prod._build_engine_behavior_contract()

        self.assertEqual(contract["signal_source"], "rule_based_setup_selector")
        self.assertIn("supported_setups", contract)
        self.assertTrue(any(item["code"] == "D" for item in contract["supported_setups"]))

    def test_compute_data_quality_distinguishes_unknown_and_legacy_labels(self):
        rows = [
            {"setup": None, "stop_loss": 1.0, "take_profit": 2.0, "close_reason": "stop_loss"},
            {"setup": "LEGACY_UNLABELED", "stop_loss": 1.0, "take_profit": 2.0, "close_reason": "take_profit"},
            {"setup": "Setup D", "stop_loss": 1.0, "take_profit": 2.0, "close_reason": "take_profit"},
            {"setup": "Setup E", "stop_loss": 1.0, "take_profit": 2.0, "close_reason": "take_profit"},
        ]

        quality = prod._compute_data_quality(rows)

        self.assertEqual(quality["unknown_setup_trades"], 1)
        self.assertEqual(quality["legacy_unlabeled_setup_trades"], 1)
        self.assertEqual(quality["noncanonical_setup_trades"], 3)

    def test_score_policy_patch_on_opportunities_counts_newly_allowed_candidates(self):
        active_policy = {
            "global": {"trade_min_rr": 1.5, "trade_min_rr_range": 1.2},
            "symbols": {"BTC": {"enabled": True}},
            "setups": {"A": {"enabled": True}, "D": {"enabled": True}},
        }
        patch = {"global": {"trade_min_rr": 1.0}}
        opportunities = [
            {
                "id": 101,
                "symbol": "BTC",
                "signal": "BUY",
                "setup": "Setup A",
                "status": "PLAN_REJECTED",
                "linked_trade_id": None,
                "planned_rr": None,
                "preview_plan_json": {
                    "status": "REJECTED",
                    "reason": "TP1 RR 1.10 < 1.50; TP2 RR 1.30 (tp1=123.0)",
                    "tp1_rr": 1.1,
                    "tp2_rr": 1.3,
                    "best_rr": 1.3,
                    "min_rr": 1.5,
                },
            }
        ]

        score = prod.score_policy_patch_on_opportunities(active_policy, patch, opportunities)

        self.assertEqual(score["newly_allowed_candidates"], 1)
        self.assertEqual(score["unscored_new_candidates"], 1)
        self.assertEqual(score["candidate"]["policy_allowed_rows"], 1)

    def test_run_policy_review_once_respects_hold_guard_without_force_review(self):
        cfg = ReviewGuardConfig(trades_table="trades_live")
        guard = ReviewGuardResult(
            decision="HOLD",
            reason="closed_trades_since_update=7 < 20",
            guard_config=asdict(cfg),
            context={
                "active_policy": {"id": 7, "version": 3},
                "policy_timing": {"hours_since_update": 8.5},
                "post_update_sample": {"closed_trades_since_update": 7},
            },
        )
        active_row = {
            "id": 7,
            "version": 3,
            "policy_json": {"global": {"trade_min_rr": 1.5}},
            "engine_name": "llm_live",
            "account_type": "live",
        }
        review_input = {
            "active_policy": {"id": 7, "version": 3},
            "guard": asdict(guard),
            "deterministic_checks": {"allow_llm_proposal": False},
        }
        captured_payload: dict[str, object] = {}

        class FakeConn:
            def commit(self):
                return None

            def close(self):
                return None

        def fake_insert(_conn, payload):
            captured_payload["payload"] = payload
            return 456

        with mock.patch.dict(
            os.environ,
            {
                "DATABASE_URL": "postgresql://example.local/test",
                "POLICY_REVIEW_ENABLED": "1",
            },
            clear=False,
        ):
            with mock.patch.object(
                prod,
                "build_policy_review_context",
                return_value=(review_input, active_row, guard, cfg),
            ) as build_ctx:
                with mock.patch.object(prod.psycopg2, "connect", return_value=FakeConn()):
                    with mock.patch.object(prod, "_insert_policy_review_run", side_effect=fake_insert):
                        with mock.patch.object(prod, "_call_llm_reviewer") as call_llm:
                            result = prod.run_policy_review_once()

        self.assertEqual(result["status"], "hold")
        self.assertEqual(result["run_id"], 456)
        self.assertEqual(result["reason"], guard.reason)
        self.assertNotIn("manual_override", result)
        self.assertFalse(call_llm.called)
        self.assertFalse(build_ctx.call_args.kwargs["bypass_maturity_gate"])
        payload = captured_payload["payload"]
        self.assertEqual(payload["guard_decision"], "HOLD")
        self.assertFalse(payload["guard_payload"]["manual_override"]["force_review"])

    def test_run_policy_review_once_force_review_bypasses_disabled_and_guard(self):
        cfg = ReviewGuardConfig(trades_table="trades_live")
        guard = ReviewGuardResult(
            decision="HOLD",
            reason="hours_since_update=2.00 < 24.00",
            guard_config=asdict(cfg),
            context={
                "active_policy": {"id": 9, "version": 4},
                "policy_timing": {"hours_since_update": 2.0},
                "post_update_sample": {"closed_trades_since_update": 1},
            },
        )
        active_row = {
            "id": 9,
            "version": 4,
            "policy_json": {"global": {"trade_min_rr": 1.5}},
            "engine_name": "llm_live",
            "account_type": "live",
        }
        review_input = {
            "active_policy": {"id": 9, "version": 4},
            "guard": asdict(guard),
            "deterministic_checks": {
                "allow_llm_proposal": True,
                "clear_signal_for_risk_increase": False,
            },
        }
        captured_payload: dict[str, object] = {}

        class FakeConn:
            def commit(self):
                return None

            def close(self):
                return None

        def fake_insert(_conn, payload):
            captured_payload["payload"] = payload
            return 123

        with mock.patch.dict(
            os.environ,
            {
                "DATABASE_URL": "postgresql://example.local/test",
                "POLICY_REVIEW_ENABLED": "0",
            },
            clear=False,
        ):
            with mock.patch.object(
                prod,
                "build_policy_review_context",
                return_value=(review_input, active_row, guard, cfg),
            ) as build_ctx:
                with mock.patch.object(prod.psycopg2, "connect", return_value=FakeConn()):
                    with mock.patch.object(prod, "_insert_policy_review_run", side_effect=fake_insert):
                        with mock.patch.object(
                            prod,
                            "_call_llm_reviewer",
                            return_value={
                                "decision": "NO_CHANGE",
                                "reason": "Manual review completed",
                                "_meta": {"model": "gpt-5-mini"},
                            },
                        ) as call_llm:
                            result = prod.run_policy_review_once(
                                force_review=True,
                                force_review_reason="telegram:/review",
                            )

        self.assertEqual(result["status"], "no_change")
        self.assertEqual(result["run_id"], 123)
        self.assertEqual(result["reason"], "Manual review completed")
        self.assertTrue(call_llm.called)
        self.assertTrue(build_ctx.call_args.kwargs["bypass_maturity_gate"])
        manual_override = result["manual_override"]
        self.assertTrue(manual_override["force_review"])
        self.assertTrue(manual_override["bypassed_policy_review_enabled"])
        self.assertTrue(manual_override["bypassed_guard_gate"])
        self.assertEqual(manual_override["reason"], "telegram:/review")
        payload = captured_payload["payload"]
        self.assertEqual(payload["llm_decision"], "NO_CHANGE")
        self.assertEqual(payload["guard_decision"], "HOLD")
        self.assertTrue(payload["guard_payload"]["manual_override"]["force_review"])
        self.assertTrue(payload["llm_payload"]["manual_override"]["bypassed_guard_gate"])
        self.assertEqual(payload["llm_payload"]["manual_override"]["reason"], "telegram:/review")

    def test_run_policy_review_once_force_review_can_auto_apply_override(self):
        cfg = ReviewGuardConfig(trades_table="trades_live")
        guard = ReviewGuardResult(
            decision="HOLD",
            reason="closed_trades_since_update=1 < 20",
            guard_config=asdict(cfg),
            context={
                "active_policy": {"id": 12, "version": 5},
                "policy_timing": {"hours_since_update": 3.0},
                "post_update_sample": {"closed_trades_since_update": 1},
            },
        )
        active_row = {
            "id": 12,
            "version": 5,
            "policy_json": {"global": {"trade_min_rr": 1.5, "trade_min_rr_range": 1.4}},
            "engine_name": "llm_live",
            "account_type": "live",
            "policy_name": "default",
        }
        review_input = {
            "active_policy": {"id": 12, "version": 5},
            "guard": asdict(guard),
            "deterministic_checks": {
                "allow_llm_proposal": True,
                "clear_signal_for_risk_increase": False,
                "risk_constraints": {"max_changed_keys": 2},
                "clear_signal_evidence": {"closed_trades_since_update": 1},
            },
        }
        captured_payload: dict[str, object] = {}

        class FakeConn:
            def commit(self):
                return None

            def close(self):
                return None

        def fake_insert(_conn, payload):
            captured_payload["payload"] = payload
            return 777

        with mock.patch.dict(
            os.environ,
            {
                "DATABASE_URL": "postgresql://example.local/test",
                "POLICY_REVIEW_ENABLED": "1",
                "POLICY_REVIEW_AUTO_APPLY": "0",
            },
            clear=False,
        ):
            with mock.patch.object(
                prod,
                "build_policy_review_context",
                return_value=(review_input, active_row, guard, cfg),
            ):
                with mock.patch.object(prod.psycopg2, "connect", return_value=FakeConn()):
                    with mock.patch.object(prod, "_insert_policy_review_run", side_effect=fake_insert):
                        with mock.patch.object(
                            prod,
                            "_call_llm_reviewer",
                            return_value={
                                "decision": "PROPOSE_CHANGE",
                                "reason": "Tighten one threshold",
                                "_meta": {"model": "gpt-5-mini"},
                                "changes": [
                                    {"path": "global.trade_min_rr_range", "value": 1.6},
                                ],
                            },
                        ):
                            with mock.patch.object(
                                prod,
                                "_create_policy_version",
                                return_value=(88, 6, "active"),
                            ) as create_version:
                                result = prod.run_policy_review_once(
                                    force_review=True,
                                    force_review_reason="telegram:/review apply",
                                    auto_apply_override=True,
                                )

        self.assertEqual(result["status"], "applied")
        self.assertEqual(result["run_id"], 777)
        self.assertEqual(result["policy_id"], 88)
        self.assertEqual(result["policy_version"], 6)
        self.assertEqual(result["policy_status"], "active")
        manual_override = result["manual_override"]
        self.assertTrue(manual_override["force_review"])
        self.assertTrue(manual_override["auto_apply_override"])
        payload = captured_payload["payload"]
        self.assertEqual(payload["llm_decision"], "APPLIED")
        self.assertTrue(payload["guard_payload"]["manual_override"]["auto_apply_override"])
        self.assertTrue(payload["llm_payload"]["manual_override"]["auto_apply_override"])
        self.assertTrue(create_version.call_args.kwargs["auto_activate"])


if __name__ == "__main__":
    unittest.main()

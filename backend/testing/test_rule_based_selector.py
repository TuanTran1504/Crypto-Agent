import sys
import unittest

import pandas as pd


sys.path.insert(0, "backend/trading")

from strategy_core import build_rule_based_signal_decision


def _df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


class RuleBasedSelectorTests(unittest.TestCase):
    def test_selector_waits_on_insufficient_candles(self):
        df_exec = _df(
            [
                {"open": 100.0, "high": 100.4, "low": 99.8, "close": 100.2, "volume": 1000},
                {"open": 100.2, "high": 100.5, "low": 100.0, "close": 100.3, "volume": 980},
                {"open": 100.3, "high": 100.7, "low": 100.1, "close": 100.6, "volume": 1100},
            ]
        )
        ctx = {
            "current_price": 100.6,
            "primary_trend": "UPTREND",
            "m15_trend": "UPTREND",
            "is_range": False,
            "allowed_direction": "BOTH",
        }

        decision = build_rule_based_signal_decision(ctx, df_exec)

        self.assertEqual(decision["signal"], "WAIT")
        self.assertIn("Not enough execution candles", decision["reason"])
        self.assertEqual(decision["_meta"]["selector"], "rule_based")

    def test_selector_produces_structured_trend_decision(self):
        df_exec = _df(
            [
                {"open": 100.8, "high": 101.0, "low": 100.7, "close": 100.9, "volume": 980},
                {"open": 100.9, "high": 101.2, "low": 100.8, "close": 101.1, "volume": 1000},
                {"open": 101.1, "high": 101.4, "low": 101.0, "close": 101.3, "volume": 990},
                {"open": 101.3, "high": 101.6, "low": 101.2, "close": 101.5, "volume": 1010},
                {"open": 101.5, "high": 101.8, "low": 101.4, "close": 101.7, "volume": 1005},
                {"open": 101.7, "high": 102.0, "low": 101.6, "close": 101.9, "volume": 1025},
                {"open": 101.9, "high": 102.1, "low": 101.8, "close": 102.0, "volume": 1030},
                {"open": 102.0, "high": 102.25, "low": 101.9, "close": 102.15, "volume": 1040},
                {"open": 102.15, "high": 103.1, "low": 102.05, "close": 103.0, "volume": 1700},
            ]
        )
        ctx = {
            "current_price": 103.0,
            "primary_trend": "UPTREND",
            "m15_trend": "UPTREND",
            "h1_trend": "UPTREND",
            "is_range": False,
            "allowed_direction": "BOTH",
            "atr_m15": 0.30,
            "atr": 0.30,
            "m15_gap": 0.85,
            "ema_gap_widening": True,
            "rsi": 64.0,
            "adx_m15": 31.0,
            "sr": {
                "support": 102.0,
                "resistance": 104.8,
                "support_levels": [102.0, 101.6, 101.1],
                "resistance_levels": [104.8, 106.0, 107.5],
            },
        }

        decision = build_rule_based_signal_decision(ctx, df_exec)

        self.assertEqual(decision["signal"], "BUY")
        self.assertIn(decision["analysis"]["setup_identified"], {"Setup A", "Setup B", "Setup C"})
        self.assertGreater(decision["planned_rr"], 0.0)
        self.assertEqual(decision["_meta"]["selector"], "rule_based")


if __name__ == "__main__":
    unittest.main()

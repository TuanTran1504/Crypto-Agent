#!/usr/bin/env bash
set -euo pipefail

# Local tester for backend/schedule/run_policy_review.py
#
# Usage:
#   ./backend/testing/test_run_policy_review_local.sh disabled
#   ./backend/testing/test_run_policy_review_local.sh live
#   ./backend/testing/test_run_policy_review_local.sh preview
#   ./backend/testing/test_run_policy_review_local.sh shadow
#   ./backend/testing/test_run_policy_review_local.sh walkforward
#
# Modes:
#   disabled -> forces POLICY_REVIEW_ENABLED=0 and prints JSON output
#   live     -> runs full review path with your local .env / DB / API keys
#   preview  -> read-only DB mode from separate testing script,
#               bypasses guard timing in prompt and calls LLM,
#               prints JSON output, writes nothing to DB
#   shadow   -> multi-agent read-only review (analyst/proposer/critic)
#   walkforward -> historical replay of the shadow reviewer plus forward scoring

MODE="${1:-disabled}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if ! command -v python >/dev/null 2>&1; then
  echo "ERROR: python is not installed."
  exit 1
fi

if [[ "$MODE" == "disabled" ]]; then
  POLICY_REVIEW_ENABLED=0 python backend/schedule/run_policy_review.py --json-only
elif [[ "$MODE" == "live" ]]; then
  python backend/schedule/run_policy_review.py --json-only
elif [[ "$MODE" == "preview" ]]; then
  POLICY_REVIEW_PREVIEW_BYPASS_TIME=1 python backend/testing/run_policy_review_read_only_llm.py --json-only
elif [[ "$MODE" == "shadow" ]]; then
  POLICY_REVIEW_PREVIEW_BYPASS_TIME=1 python backend/testing/run_policy_review_shadow_multi_agent.py --json-only
elif [[ "$MODE" == "walkforward" ]]; then
  python backend/testing/run_policy_walkforward_eval.py --json-only
else
  echo "ERROR: unknown mode '$MODE'. Use: disabled | live | preview | shadow | walkforward"
  exit 1
fi

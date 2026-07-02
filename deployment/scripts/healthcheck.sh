#!/usr/bin/env bash
# Readiness probe for the vLLM server. Exits 0 and prints "OK" if the server is up and the
# expected model is loaded; exits 1 with a diagnostic message otherwise. Suitable for use as a
# systemd ExecStartPost check, a cron-based alert, or a load-balancer health check target.
set -euo pipefail

BASE_URL="${VLLM_BASE_URL:-http://localhost:8000}"
EXPECTED_MODEL="${VLLM_MODEL_NAME:-qwen3-8b-vedaz}"
TIMEOUT="${HEALTHCHECK_TIMEOUT:-5}"

if ! curl -sf --max-time "$TIMEOUT" "$BASE_URL/health" > /dev/null; then
    echo "FAIL: $BASE_URL/health did not respond within ${TIMEOUT}s"
    exit 1
fi

models_response=$(curl -sf --max-time "$TIMEOUT" "$BASE_URL/v1/models" || true)
if ! echo "$models_response" | grep -q "$EXPECTED_MODEL"; then
    echo "FAIL: expected model '$EXPECTED_MODEL' not found in /v1/models response: $models_response"
    exit 1
fi

echo "OK: $BASE_URL is healthy and serving '$EXPECTED_MODEL'"
exit 0

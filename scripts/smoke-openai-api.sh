#!/usr/bin/env bash
set -euo pipefail

base_url="${MINI_OLLAMA_URL:-http://127.0.0.1:11434}"
model_name="${1:-}"
work_file="$(mktemp)"
trap 'rm -f "$work_file"' EXIT

curl --fail --silent --show-error "$base_url/v1/models" >"$work_file"
python - "$work_file" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    payload = json.load(stream)
if payload.get("object") != "list" or not isinstance(payload.get("data"), list):
    raise SystemExit("invalid /v1/models response")
print(f"models: {len(payload['data'])}")
PY

missing_code="$(curl --silent --output /dev/null --write-out '%{http_code}' \
  -X POST "$base_url/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"__mini_ollama_missing__","messages":[{"role":"user","content":"ping"}]}' )"
if [[ "$missing_code" != "404" ]]; then
  echo "expected missing-model HTTP 404, got $missing_code" >&2
  exit 1
fi

echo "missing-model error: HTTP $missing_code"

status_code="$(curl --silent --output /dev/null --write-out '%{http_code}' \
  "$base_url/api/v1/metrics" || true)"
case "$status_code" in
  200|503) echo "metrics endpoint: HTTP $status_code" ;;
  *) echo "unexpected metrics HTTP $status_code" >&2; exit 1 ;;
esac

if [[ "${SMOKE_READY:-0}" != "1" ]]; then
  echo "routing and error smoke checks passed"
  exit 0
fi

if [[ -z "$model_name" ]]; then
  echo "SMOKE_READY=1 requires a model name argument" >&2
  exit 2
fi

curl --fail --silent --show-error \
  -X POST "$base_url/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$model_name\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with one short sentence.\"}],\"stream\":false,\"max_tokens\":32}" \
  >"$work_file"
python - "$work_file" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    payload = json.load(stream)
if payload.get("object") != "chat.completion":
    raise SystemExit("invalid non-stream chat completion")
if not payload.get("choices"):
    raise SystemExit("chat completion has no choices")
print("non-stream completion: ok")
PY

curl --fail --silent --show-error -N \
  -X POST "$base_url/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$model_name\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with one short sentence.\"}],\"stream\":true,\"max_tokens\":32}" \
  | tee "$work_file" >/dev/null

grep -q 'chat.completion.chunk' "$work_file"
grep -q 'data: \[DONE\]' "$work_file"
echo "stream completion: ok"
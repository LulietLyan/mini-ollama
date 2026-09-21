#!/usr/bin/env bash
set -euo pipefail

# The work host keeps its required network proxy setup in ~/.bashrc.
if [[ -f "${HOME}/.bashrc" ]]; then
  # shellcheck disable=SC1090
  source "${HOME}/.bashrc"
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_DIR="${ROOT_DIR}/third_party/llama.cpp"
LLAMA_REPO="https://github.com/ggml-org/llama.cpp.git"
LLAMA_COMMIT="$(cat "${ROOT_DIR}/ollama/LLAMA_CPP_VERSION" 2>/dev/null || git -C "${ROOT_DIR}/ollama" show HEAD:LLAMA_CPP_VERSION)"

mkdir -p "${ROOT_DIR}/third_party"

if [[ ! -d "${LLAMA_DIR}/.git" ]]; then
  if [[ -e "${LLAMA_DIR}" ]]; then
    echo "third_party/llama.cpp exists but is not a git checkout: ${LLAMA_DIR}" >&2
    exit 1
  fi
  git clone "${LLAMA_REPO}" "${LLAMA_DIR}"
fi

expected_full=""
if expected_full="$(git -C "${LLAMA_DIR}" rev-parse --verify "${LLAMA_COMMIT}^{commit}" 2>/dev/null)"; then
  :
fi
if [[ -z "${expected_full}" ]]; then
  git -C "${LLAMA_DIR}" fetch --depth 1 origin "${LLAMA_COMMIT}"
  expected_full="$(git -C "${LLAMA_DIR}" rev-parse FETCH_HEAD)"
fi

current_commit="$(git -C "${LLAMA_DIR}" rev-parse HEAD 2>/dev/null || true)"
if [[ "${current_commit}" != "${expected_full}" ]]; then
  git -C "${LLAMA_DIR}" checkout --detach "${expected_full}"
fi

actual_commit="$(git -C "${LLAMA_DIR}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${expected_full}" ]]; then
  echo "llama.cpp checkout mismatch: expected ${expected_full} (${LLAMA_COMMIT}), got ${actual_commit}" >&2
  exit 1
fi
echo "llama.cpp ready at ${actual_commit} (${LLAMA_DIR})"

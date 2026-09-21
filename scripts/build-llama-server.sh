#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="${ROOT_DIR}/third_party/llama.cpp"
BUILD_DIR="${ROOT_DIR}/build/llama-server"

if [[ ! -d "${SRC_DIR}/.git" ]]; then
  echo "llama.cpp is missing; run scripts/bootstrap.sh first" >&2
  exit 1
fi

if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
  cmake -S "${SRC_DIR}" -B "${BUILD_DIR}" -G Ninja \
    -DGGML_METAL=ON \
    -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_CURL=OFF \
    -DCMAKE_BUILD_TYPE=Release
else
  if [[ "$(uname -s)" != "Linux" ]]; then
    echo "Unsupported platform: $(uname -s)" >&2
    exit 1
  fi
  NVCC="$(command -v nvcc 2>/dev/null || true)"
  if [[ -z "${NVCC}" && -x /usr/local/cuda/bin/nvcc ]]; then
    NVCC=/usr/local/cuda/bin/nvcc
  fi
  if [[ -z "${NVCC}" ]]; then
    echo "nvcc not found (checked PATH and /usr/local/cuda/bin/nvcc); CUDA build cannot continue" >&2
    exit 1
  fi
  CUDA_BIN_DIR="$(cd "$(dirname "${NVCC}")" && pwd)"
  export PATH="${CUDA_BIN_DIR}:${PATH}"
  CUDA_HOST_CXX=""
  for candidate in /usr/bin/g++-13 /usr/bin/g++-12 /usr/bin/g++-11 /usr/bin/g++-10 /usr/bin/g++-9 /usr/bin/g++; do
    if [[ -x "${candidate}" ]]; then
      CUDA_HOST_CXX="${candidate}"
      break
    fi
  done
  if [[ -z "${CUDA_HOST_CXX}" ]]; then
    echo "No compatible host g++ found for CUDA nvcc; install one through conda" >&2
    exit 1
  fi
  CUDA_HOST_C="${CUDA_HOST_CXX/g++/gcc}"
  cmake -S "${SRC_DIR}" -B "${BUILD_DIR}" -G Ninja \
    -DCMAKE_C_COMPILER="${CUDA_HOST_C}" \
    -DCMAKE_CXX_COMPILER="${CUDA_HOST_CXX}" \
    -DCMAKE_CUDA_HOST_COMPILER="${CUDA_HOST_CXX}" \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=86 \
    -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_CURL=OFF \
    -DCMAKE_BUILD_TYPE=Release
fi

cmake --build "${BUILD_DIR}" --target llama-server --parallel "${JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu)}"
mkdir -p "${BUILD_DIR}/bin"
if [[ -x "${BUILD_DIR}/bin/llama-server" ]]; then
  :
elif [[ -x "${BUILD_DIR}/bin/server" ]]; then
  cp "${BUILD_DIR}/bin/server" "${BUILD_DIR}/bin/llama-server"
else
  found="$(find "${BUILD_DIR}" -type f -name llama-server -perm -111 -print -quit)"
  [[ -n "${found}" ]] || { echo "llama-server executable was not produced" >&2; exit 1; }
  cp "${found}" "${BUILD_DIR}/bin/llama-server"
fi
echo "Built ${BUILD_DIR}/bin/llama-server"

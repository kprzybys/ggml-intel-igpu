#!/bin/sh
# Do the spare packed buffers (lib/gen9blas.c packed_buf, GEN9BLAS_SPARE_MB) make the GEN9 backend hold more memory than
# GEN9_CACHE_MB + the spare cap? Risk from the 2026-11-13 review: a small weight that fits the cache budget takes a large
# spare buffer (e.g. a 136 MB FFN buffer for a 3.7 MB k/v weight) and keeps it for good, so real memory exceeds the budget.
# Runs llama-bench pp512 with a cache budget that leaves part of the weights packed per use, with the pool off and on,
# and compares the peak RSS. Fails if the difference is larger than the spare cap + 64 MB.
# Usage: bench/check_spare_budget.sh [model.gguf] [GEN9_CACHE_MB] [GEN9BLAS_SPARE_MB]
#   defaults: Qwen2.5-1.5B Q4_K_M, 1500, 512 (7B: ~/models/qwen2.5-7b/Qwen2.5-7B-Instruct-Q4_K_M.gguf 4000 512)
set -e
repo=$(cd -P "$(dirname "$0")/.." && pwd)
m=${1:-$HOME/models/qwen2.5-1.5b/qwen2.5-1.5b-instruct-q4_k_m.gguf}; budget=${2:-1500}; cap=${3:-512}
bench=${LLAMA_BENCH:-$HOME/src/llama.cpp/build-cpu/bin/llama-bench}
peak() {   # $1 = spare MB -> "peak_RSS_MB per_use_packs t/s"
  out=$(GEN9BLAS_SPARE_MB=$1 GEN9_CACHE_MB=$budget GEN9_INFO=1 GGML_BACKEND_PATH="$repo/ggml-backend/libggml-gen9.so" \
        /usr/bin/time -v "$bench" -m "$m" -p 512 -n 0 -r 2 2>&1)
  rss=$(echo "$out" | awk -F: '/Maximum resident set size/ {print int($2 / 1024)}')
  pu=$(echo "$out" | grep -o 'packed per use [0-9]*' | awk '{print $4}')
  ts=$(echo "$out" | grep -oE '\| *pp512 *\| *[0-9.]+' | awk '{print $NF}')
  echo "$rss ${pu:-?} ${ts:-?}"
}
set -- $(peak 0); r0=$1; p0=$2; t0=$3
set -- $(peak "$cap"); r1=$1; p1=$2; t1=$3
d=$((r1 - r0))
echo "model $(basename "$m"), budget $budget MB, spare cap $cap MB"
echo "pool off: peak RSS $r0 MB, per-use packs $p0, pp512 $t0 t/s"
echo "pool on : peak RSS $r1 MB, per-use packs $p1, pp512 $t1 t/s"
echo "difference $d MB (limit $((cap + 64)) MB)"
[ "$p0" != 0 ] || echo "warning: no per-use packing at this budget; the test says nothing (lower GEN9_CACHE_MB)"
if [ "$d" -le $((cap + 64)) ]; then echo "OK"; else echo "FAIL: the pool holds more than its cap"; exit 1; fi

#!/bin/bash
# whisper_dec_ab.sh -- decoder of whisper on the GPU (M = 1 MUL_MAT) vs CPU, no code change: GEN9_MIN_BATCH 8 vs 1.
# Model and falsification criteria: results/whisper-decoder-gpu-model-2026-11-13.txt. Runs on gp with a separate run's
# whisper-cli build (read-only use) and the GEN9 backend .so given as $1. Alternating A B A B, separate processes.
# Usage (env WHISPER, MODEL, OUT override the defaults): bash whisper_dec_ab.sh <libggml-gen9-*.so> [wav=~/data/fleurs-pl/pl60.wav] [seconds=60]
set -u
SO=$1; SRC=${2:-$HOME/data/fleurs-pl/pl60.wav}; SECS=${3:-60}
W=${WHISPER:-$HOME/src/whisper.cpp}; M=${MODEL:-$HOME/models/whisper/ggml-large-v3-turbo.bin}; D=${OUT:-$PWD/dec-ab}
mkdir -p "$D"
for f in /sys/class/drm/card*/engine/*/preempt_timeout_ms; do
    v=$(cat "$f"); echo "# $f = $v"; [ "$v" = 640 ] || { echo "ABORT: $f is $v"; exit 2; }
done
echo "# host $(hostname), $(date -Is), load $(cut -d' ' -f1-3 /proc/loadavg)"
echo "# whisper.cpp $(git -C "$W" rev-parse --short HEAD), backend $SO ($(sha256sum "$SO" | cut -c1-16)), -t 4 -bs 1 -bo 1, GEN9_ATTN=1 GEN9_EWOPS=1 (with MIN_BATCH 1 the elementwise ops also go to the GPU: best case for the GPU)"
strings "$SO" | grep -q GEN9_MIN_BATCH || { echo "ABORT: backend does not read GEN9_MIN_BATCH"; exit 3; }
python3 -I - "$SRC" "$D/clip.wav" "$SECS" <<'P'
import sys, wave
src, dst, secs = sys.argv[1], sys.argv[2], float(sys.argv[3])
with wave.open(src) as r:
    p = r.getparams(); frames = r.readframes(int(secs * p.framerate))
with wave.open(dst, "wb") as w:
    w.setparams(p); w.writeframes(frames)
print(f"# clip: {src} first {secs:.0f} s, {p.framerate} Hz, {p.nchannels} ch")
P
run() {  # $1 = GEN9_MIN_BATCH, $2 = round
    local out="$D/mb$1-r$2"
    env GGML_BACKEND_PATH="$SO" GEN9_INFO=1 GEN9_ATTN=1 GEN9_EWOPS=1 GEN9_MIN_BATCH=$1 nice -n 0 "$W/build-cpu/bin/whisper-cli" \
        -m "$M" -f "$D/clip.wav" -l pl -t 4 -bs 1 -bo 1 -otxt -of "$out" > "$out.log" 2>&1
    echo "## GEN9_MIN_BATCH=$1 round $2 (exit $?)"
    grep -E "decode time|encode time|batchd time|prompt time|total time|fallbacks" "$out.log"
    grep -iE "MUL_MAT|declin|GPU init|error" "$out.log" | head -5
}
for r in 1 2; do run 8 $r; run 1 $r; done
echo "## transcripts identical? $(cmp -s "$D/mb8-r1.txt" "$D/mb1-r1.txt" && echo yes || echo NO)"
echo "# done $(date -Is)"

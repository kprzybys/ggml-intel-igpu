"""One generated token of Qwen2.5-1.5B as 197 FP16 GEMVs (M = 1) with the existing skinny_k1 kernel, one submission each.
Question (step 2 of docs/ggml-backend-design.md): is the GPU token limited by the kernels or by the submissions?
Reference: llama.cpp CPU tg F16 6.9 t/s = 145 ms/token (results/llama-cpp-openvino-2026-11-13.txt).

Run: python3 bench/token_gemv.py      (3 rounds; plain and KCL_PROFILE_KERNELS=1 processes alternate)
Shapes (ggml src0 = N x K, weights contiguous along K): per layer Q 1536x1536, K 256x1536, V 256x1536, O 1536x1536,
gate 8960x1536, up 8960x1536, down 1536x8960; 28 layers + head 151936x1536. 3.09 GB of FP16 weights in USM.
"""
import os
import re
import subprocess
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np  # noqa: E402

H, I, KV, L, V = 1536, 8960, 256, 28, 151936
LAYER = [("q", H, H), ("k", KV, H), ("v", KV, H), ("o", H, H), ("gate", I, H), ("up", I, H), ("down", H, I)]
SHAPES = [s for _ in range(L) for s in LAYER] + [("head", V, H)]
BYTES = sum(n * k * 2 for _, n, k in SHAPES)
WARM, TOKENS = 2, 10


def child():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "lib"))
    import gen9blas
    Ws, xs, ys = [], {}, {}
    for _, n, k in SHAPES:
        W = gen9blas.empty((n, k), np.float16); W[...] = 1e-3; Ws.append(W)
        if k not in xs:
            xs[k] = gen9blas.empty((1, k), np.float16); xs[k][...] = 1
        if (n, k) not in ys:
            ys[n, k] = gen9blas.empty((1, n), np.float16)

    def token():
        for W, (_, n, k) in zip(Ws, SHAPES):
            gen9blas.hgemm(xs[k], W.T, C=ys[n, k])

    for _ in range(WARM):
        token()
    walls, cpus = [], []
    for _ in range(TOKENS):
        c, t = time.process_time(), time.perf_counter()
        token()
        walls.append(time.perf_counter() - t); cpus.append(time.process_time() - c)
    for (n, k), y in ys.items():
        assert abs(float(y[0, 0]) - k * 1e-3) < 0.01 * k * 1e-3, (n, k, float(y[0, 0]))
    print(f"WALL {np.median(walls) * 1e3:.2f} {min(walls) * 1e3:.2f} CPU {np.median(cpus) * 1e3:.2f}", flush=True)


def run(profile):
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1")
    if profile:
        env["KCL_PROFILE_KERNELS"] = "1"
    p = subprocess.run([sys.executable, __file__, "-"], env=env, capture_output=True, text=True, check=True)
    m = re.search(r"WALL (\S+) (\S+) CPU (\S+)", p.stdout)
    wall, wmin, cpu = map(float, m.groups())
    kern = None
    if profile:
        k = re.search(r"kcl: skinny_k1\w*\s+(\d+) x\s+(\S+) ms", p.stderr)
        n, ms = int(k.group(1)), float(k.group(2))
        kern = ms / (n / len(SHAPES))   # all tokens (warm-up included) launch the same 197 kernels
        assert n == len(SHAPES) * (WARM + TOKENS), (n, p.stderr)
    return wall, wmin, cpu, kern


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "-":
        child(); sys.exit()
    print(f"# {len(SHAPES)} GEMVs per token, {BYTES / 1e9:.3f} GB FP16 weights; power online="
          f"{open([f'/sys/class/power_supply/{d}/online' for d in os.listdir('/sys/class/power_supply') if d.startswith('A')][0]).read().strip()}")
    print("round mode     wall_med  wall_min  cpu_med  kernels  [ms/token]   GB/s(kernels)")
    plain, prof = [], []
    for r in range(3):
        for profile in ((False, True) if r % 2 == 0 else (True, False)):
            wall, wmin, cpu, kern = run(profile)
            (prof if profile else plain).append((wall, cpu, kern))
            ks = f"{kern:8.1f}  {BYTES / kern / 1e6:6.1f}" if kern else ""
            print(f"{r}     {'profile' if profile else 'plain  '}  {wall:8.1f}  {wmin:8.1f}  {cpu:7.1f}  {ks}", flush=True)
    w = np.median([p[0] for p in plain]); c = np.median([p[1] for p in plain]); k = np.median([p[2] for p in prof])
    print(f"\nmedian: wall {w:.1f} ms/token ({1e3 / w:.2f} t/s), CPU {c:.1f} ms, kernels {k:.1f} ms "
          f"({BYTES / k / 1e6:.1f} GB/s), wall - kernels {w - k:.1f} ms = {(w - k) / len(SHAPES) * 1e3:.0f} us per submission")
    print("decision (from the review): kernels > 125 ms -> no kernel beats the CPU (145 ms); kernels <= 125 and wall > 145 "
          "-> work on submissions; otherwise -> microbenchmark gemv_pb")

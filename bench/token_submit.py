"""Follow-up to bench/token_gemv.py (2026-11-13): the same token (197 FP16 GEMVs of Qwen2.5-1.5B, skinny_k1, USM), but
with fewer waits for the GPU. How much of the ~33 ms submission overhead per token goes away?

Modes (all in one process, rotated each round):
  per_gemv  -- wait after every GEMV (197 waits, as today)
  per_split -- wait once per ggml split: per layer Q; K; V; O; gate+up; down (bias ADDs on the CPU separate Q/K/V,
               docs/ggml-backend-design.md 6.2) + head = 169 waits
  per_layer -- wait once per layer (as if the CPU ops between MUL_MATs moved to the GPU) = 29 waits
  one       -- one wait per token (lower bound; kcl still submits a batch buffer every 64 kernels)
The deferral is an LD_PRELOAD shim over kcl_wait (bench/defer_wait.c), so the library is not changed.

Run: gcc -O2 -shared -fPIC -o bench/defer_wait.so bench/defer_wait.c -ldl && python3 bench/token_submit.py
"""
import ctypes
import os
import subprocess
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from token_gemv import SHAPES, BYTES, L  # noqa: E402

SHIM = os.path.join(os.path.dirname(os.path.realpath(__file__)), "defer_wait.so")
ROUNDS, TOKENS = 5, 5


def groups(mode):
    idx = list(range(len(SHAPES)))
    if mode == "per_gemv":
        return [[i] for i in idx]
    if mode == "one":
        return [idx]
    out = []
    for layer in range(L):
        b = 7 * layer
        out += [[b], [b + 1], [b + 2], [b + 3], [b + 4, b + 5], [b + 6]] if mode == "per_split" else [list(range(b, b + 7))]
    return out + [[len(SHAPES) - 1]]


def child():
    shim = ctypes.CDLL(SHIM)
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "lib"))
    import gen9blas
    shim.defer_wait_init(ctypes.cast(gen9blas._lib.kcl_wait, ctypes.c_void_p))   # before the library's first wait
    Ws, xs, ys = [], {}, {}
    for _, n, k in SHAPES:
        W = gen9blas.empty((n, k), np.float16); W[...] = 1e-3; Ws.append(W)
        if k not in xs:
            xs[k] = gen9blas.empty((1, k), np.float16); xs[k][...] = 1
        if (n, k) not in ys:
            ys[n, k] = gen9blas.empty((1, n), np.float16)

    def token(gs):
        for g in gs:
            shim.defer_wait_set(1)
            for i in g:
                _, n, k = SHAPES[i]
                gen9blas.hgemm(xs[k], Ws[i].T, C=ys[n, k])
            shim.defer_wait_set(0)
            assert shim.defer_wait_sync() == 0

    modes = ["per_gemv", "per_split", "per_layer", "one"]
    G = {m: groups(m) for m in modes}
    for m in modes:
        token(G[m])
    # the shim is active: a deferred token returns long before the GPU work is done
    shim.defer_wait_set(1); t = time.perf_counter()
    for i, (_, n, k) in enumerate(SHAPES):
        gen9blas.hgemm(xs[k], Ws[i].T, C=ys[n, k])
    t_enq = time.perf_counter() - t; shim.defer_wait_set(0); shim.defer_wait_sync()
    print(f"ENQ {t_enq * 1e3:.2f}", flush=True)
    for r in range(ROUNDS):
        for m in modes[r % 4:] + modes[:r % 4]:
            for _ in range(TOKENS):
                for y in ys.values():
                    y[...] = 0
                c, t = time.process_time(), time.perf_counter()
                token(G[m])
                print(f"T {m} {(time.perf_counter() - t) * 1e3:.2f} {(time.process_time() - c) * 1e3:.2f}", flush=True)
                for (n, k), y in ys.items():
                    assert abs(float(y[0, -1]) - k * 1e-3) < 0.01 * k * 1e-3, (m, n, k, float(y[0, -1]))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "-":
        child(); sys.exit()
    if subprocess.run(["pgrep", "^llama-"], capture_output=True).returncode == 0:
        sys.exit("a llama-* process is running -- the GPU and RAM are shared, results would be disturbed")
    env = dict(os.environ, LD_PRELOAD=SHIM, OPENBLAS_NUM_THREADS="1")
    out = subprocess.run([sys.executable, __file__, "-"], env=env, capture_output=True, text=True, check=True).stdout
    res = {}
    for line in out.splitlines():
        p = line.split()
        if p[0] == "ENQ":
            print(f"# {len(SHAPES)} GEMVs, {BYTES / 1e9:.3f} GB FP16; deferred enqueue of a whole token: {p[1]} ms (shim active)")
        else:
            res.setdefault(p[1], []).append((float(p[2]), float(p[3])))
    print(f"# {ROUNDS} rounds x {TOKENS} tokens, modes rotated; reference: kernels 117 ms (token-gemv-2026-11-13-quiet.txt), CPU llama.cpp 145 ms")
    print(f"{'mode':10s} {'waits':>5s}  wall_med  wall_min  cpu_med  [ms/token]   t/s   overhead/wait vs 117 ms")
    for m, v in res.items():
        w = np.array([a for a, _ in v]); c = np.array([b for _, b in v]); nw = len(groups(m))
        print(f"{m:10s} {nw:5d}  {np.median(w):8.1f}  {w.min():8.1f}  {np.median(c):7.1f}   {1e3 / np.median(w):5.2f}   "
              f"{(np.median(w) - 117) / nw * 1e3:6.0f} us")

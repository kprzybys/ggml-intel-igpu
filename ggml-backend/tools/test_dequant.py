"""GPU dequantization (gen9blas_pack_b_q) against ggml's own dequantize_row_* + ggml_fp32_to_fp16_row, bit for bit.

Run: python3 ggml-backend/tools/test_dequant.py [path/to/libggml-base.so]
Random block bytes (scales, quants) with realistic FP16 block scales d / dmin; the packed result is read back through
gen9blas_gemm_pb_f16w with A = identity (every product is w * 1, sums with zeros, so C = W^T exactly, except that
-0 reads back as +0). Also: FP16
range flag, padding rows (N not a multiple of 16), and the time of a Qwen2.5-1.5B-sized FFN weight vs the CPU path.
"""
import ctypes
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "lib"))
import gen9blas  # noqa: E402

GG = ctypes.CDLL(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.environ.get("LLAMA_BUILD", os.path.expanduser("~/src/llama.cpp/build-cpu")), "bin", "libggml-base.so"))
L = gen9blas._lib
_i, _p = ctypes.c_int, ctypes.c_void_p
L.gen9blas_pack_b_q.argtypes = [_i, _i, _i, _p, ctypes.c_size_t, ctypes.POINTER(_i)]
L.gen9blas_pack_b_q.restype = _p
L.gen9blas_gemm_pb_f16w.argtypes = [_i, _p, _i, _p, _p, _i, ctypes.POINTER(_i)]
L.gen9blas_packed_free.argtypes = [_p]
GG.ggml_fp32_to_fp16_row.argtypes = [_p, _p, ctypes.c_int64]
gen9blas._init()
rng = np.random.default_rng(11)
errors = 0
# qtype: (name, values per block, bytes per block, offsets of FP16 scale fields, ggml function)
Q = {0: ("Q8_0", 32, 34, [0], GG.dequantize_row_q8_0),
     1: ("Q4_K", 256, 144, [0, 2], GG.dequantize_row_q4_K),
     2: ("Q6_K", 256, 210, [208], GG.dequantize_row_q6_K)}
for q in Q.values():
    q[4].argtypes = [_p, _p, ctypes.c_int64]


def blocks(qt, K, N, d_range):
    _, qb, bb, offs, _ = Q[qt]
    W = rng.integers(0, 256, size=(N, K // qb * bb), dtype=np.uint8)
    v = W.reshape(N, K // qb, bb)
    for o in offs:
        d = rng.uniform(*d_range, size=(N, K // qb)).astype(np.float16)
        v[:, :, o:o + 2] = d.view(np.uint8).reshape(N, K // qb, 2)
    return W


def ggml_ref(qt, W, K, N):
    f = np.empty((N, K), np.float32)
    for n in range(N):
        Q[qt][4](W[n].ctypes.data, f[n].ctypes.data, K)
    h = np.empty((N, K), np.float16)
    GG.ggml_fp32_to_fp16_row(f.ctypes.data, h.ctypes.data, N * K)
    return f, h


def gpu_pack(qt, W, K, N):
    oor = _i(0)
    P = L.gen9blas_pack_b_q(qt, K, N, W.ctypes.data, W.shape[1], ctypes.byref(oor))
    if not P:
        raise RuntimeError(L.gen9blas_error().decode())
    return P, oor.value


def readback(P, K, N):
    A = np.eye(K, dtype=np.float32)
    C = np.empty((K, N), np.float32)
    nf = _i(0)
    assert L.gen9blas_gemm_pb_f16w(K, A.ctypes.data, K, P, C.ctypes.data, N, ctypes.byref(nf)) == 0
    return C.T


def case(qt, K, N, d_range=(0.001, 0.05), expect_oor=False):
    global errors
    W = blocks(qt, K, N, d_range)
    f, h = ggml_ref(qt, W, K, N)
    P, oor = gpu_pack(qt, W, K, N)
    ref_oor = not np.all(np.abs(f) <= 65504.0)
    name = f"{Q[qt][0]} K={K} N={N} d in {d_range}"
    if oor != ref_oor or ref_oor != expect_oor:
        print(f"FAIL {name}: flag {oor}, reference {ref_oor}, expected {expect_oor}"); errors += 1
    elif not oor:
        G = readback(P, K, N).astype(np.float16)
        hz = np.where(h == 0, np.float16(0), h)   # readback through the GEMM turns -0 into +0 (1 * -0 + 0)
        diff = np.count_nonzero(G.view(np.uint16) != hz.view(np.uint16))
        sub = np.count_nonzero((np.abs(h) < 6.104e-5) & (h != 0))
        print(f"{'ok  ' if diff == 0 else 'FAIL'} {name}: {diff} of {N * K} FP16 values differ from ggml (subnormal in ref: {sub})")
        errors += diff != 0
    else:
        print(f"ok   {name}: out-of-range flag set as in the reference")
    L.gen9blas_packed_free(P)


for qt in (0, 1, 2):
    case(qt, 512, 37)
    case(qt, 1536, 160)
    case(qt, 1536, 4096)   # a first Q4_K kernel was wrong only from N ~ 1024 up
    case(qt, 256 if qt else 64, 16, d_range=(0.00001, 0.0002))   # many subnormal FP16 results
case(0, 64, 16, d_range=(600.0, 1000.0), expect_oor=True)        # 127 * 1000 > 65504
case(2, 256, 16, d_range=(30.0, 60.0), expect_oor=True)          # 60 * 127 * 32 > 65504

# timing: one Qwen2.5-1.5B FFN gate weight (8960 x 1536)
for qt in (1, 2, 0):
    K, N = 1536, 8960
    W = blocks(qt, K, N, (0.001, 0.05))
    t0 = time.perf_counter(); P, _ = gpu_pack(qt, W, K, N); t1 = time.perf_counter()
    L.gen9blas_packed_free(P)
    t2 = time.perf_counter(); P, _ = gpu_pack(qt, W, K, N); t3 = time.perf_counter()
    L.gen9blas_packed_free(P)
    t4 = time.perf_counter(); ggml_ref(qt, W[:1024], K, 1024); t5 = time.perf_counter()
    print(f"     {Q[qt][0]} {N}x{K}: GPU dequant+pack {1e3 * (t1 - t0):.1f} / {1e3 * (t3 - t2):.1f} ms "
          f"({N * K * 2 / (t3 - t2) / 1e9:.1f} GB/s FP16 out); ggml CPU 1 thread {1e3 * (t5 - t4) * N / 1024:.0f} ms")
print("errors:", errors)
sys.exit(1 if errors else 0)

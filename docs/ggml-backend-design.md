# Design: a ggml backend on gen9blas kernels (draft 3, after two reviews)

State 2026-11-12. Nothing of this is implemented. Written before any code, following the project method: model and
numeric predictions first, then one variant and a measurement ([performance-model.md](performance-model.md)).
Evidence tags as elsewhere: [MEAS] measured here, [SRC] read in source code (path:line, llama.cpp commit 2cbe0ec),
[HYP] hypothesis to be tested. Drafts 1 and 2 were reviewed; the changes are listed in sections 10 and 11.

## 1. Why

`results/llama-cpp-blas-2026-11-12.txt` measured llama.cpp (Qwen2.5-1.5B-Instruct, F32, pp512) with gen9blas injected
under ggml's BLAS backend (`LD_PRELOAD`). Absolute speeds drifted between sessions (gen9blas 62.7 t/s before a reboot,
66 right after it, 81-82 half an hour later, same command [MEAS]), so only back-to-back runs compare:

| Comparison (same session, alternating) | t/s | Process CPU |
|---|---|---|
| session A (IOMMU on): OpenBLAS / gen9blas / CLBlast shim / Vulkan `-ngl 99` | 80.8 / 62.7 / 39.2 / 37.1 | 510 / 180 / 125 / 21 % |
| session B (IOMMU off, later): OpenBLAS / gen9blas | 66.1-66.3 / 81.2-81.5 | gen9blas about 1/3 of OpenBLAS |

`llama-perplexity` on the Vulkan backend hung the GPU; gen9blas and CLBlast passed with the same perplexity as the CPU
[MEAS]. One pp512 pass with gen9blas (session B, 6.08 s at 84 t/s) splits into [MEAS]:

| Part | s per pass | Share |
|---|---|---|
| GEMM kernel `sgemm_v` (1299 GFLOP, 365 GFLOPS) | 3.56 | 59 % |
| packing the transposed weights `pkf_b_t` | 1.26 | 21 % |
| ggml-cpu ops outside BLAS | 0.85 | 14 % |
| execbuf, mostly userptr page pinning | 0.20 | 3 % |
| packing activations `pkf_a` | 0.14 | 2 % |

The BLAS interface is stateless, so the library pins and re-packs the same weights on every call (193 calls per pass).
A backend can pack each weight once. Token generation (batch 1) never reaches ggml-blas (`min_batch = 32`,
`ggml-blas.cpp:420`) [SRC]; it runs on the CPU at 2.28 t/s, Vulkan reaches 3.87 t/s [MEAS].

Goal of the first version: MUL_MAT with F32 and F16 weights on the GPU from weights packed once; everything else on
ggml-cpu. Not yet: quantized types, attention, other ops, multi-GPU, training.

## 2. How ggml dispatches work (facts the design relies on)

- **Backend shape.** A backend is a `ggml_backend_reg` (`api_version = GGML_BACKEND_API_VERSION` = 3) with devices; a
  device has `supports_op`, `supports_buft`, optional `offload_op`, a default buffer type and `init_backend`; the
  backend has `graph_compute` (`ggml-backend-impl.h`) [SRC]. Every async / event / plan entry may be NULL with a
  synchronous `graph_compute`: the scheduler falls back to `synchronize` and sync copies (`ggml-backend.cpp:467-471,
  2115-2128, 2138`) [SRC]. `ggml-blas.cpp` (532 lines) is a complete example.
- **How ggml-blas gets its work without owning any weight.** Its default buffer type is the CPU buffer type
  (`ggml-blas.cpp:383-384`) and `supports_buft` accepts any host buffer (`ggml-blas.cpp:452-453`) [SRC]. llama.cpp does
  not add that buffer type to the weight list (`if (buft != ggml_backend_cpu_buffer_type())`, `llama-model.cpp:1239`)
  [SRC], so weights stay in ordinary (mmap) CPU buffers. llama.cpp creates a backend for every ACCEL device and puts it
  before the CPU backend (`llama-context.cpp:487-499`) [SRC]; the scheduler gives a node to the first backend that
  accepts the weight's buffer type and the op (`ggml-backend.cpp:1311-1316`) [SRC]. So ggml-blas runs every MUL_MAT it
  accepts, reading the weights in place. (ggml-zdnn instead returns its own buffer type and receives the weights.)
- **Weight buffers are marked.** llama.cpp sets `GGML_BACKEND_BUFFER_USAGE_WEIGHTS` on its model buffers
  (`llama-model.cpp:2098, 2409, 2463`) [SRC]; the last layer of Qwen2 keeps only the output rows before the FFN
  (`ggml_get_rows(..., inp_out_ids)`, `src/models/qwen2.cpp:97-99`) [SRC], which is why a llama-bench pp512 pass has
  193 BLAS calls instead of 196 (54 + 27 FFN calls) [MEAS for the counts].
- **Host tensors can change without `set_tensor`.** Copies between backends write `dst->data` directly when the buffer
  is host (`ggml-backend.cpp:839-842`), CPU ops write in place, and the loader without mmap reads straight into
  `cur->data` (`llama-model-loader.cpp:1836-1838`) [SRC]. ggml-zdnn therefore declines weights that are computed,
  inputs, views or trainable (`ggml-zdnn.cpp:126-148`) and resets its copy in `set_tensor` / `memset_tensor` / `clear`
  (`ggml-zdnn.cpp:359-411`) [SRC].
- **Out-of-tree loading.** `ggml_backend_load_all()` loads `$GGML_BACKEND_PATH` unconditionally
  (`ggml-backend-reg.cpp:594-598`) [SRC]. The library must export `ggml_backend_init` through
  `GGML_BACKEND_DL_IMPL(reg_fn)`, which is **empty unless the backend is compiled with `GGML_BACKEND_DL` and
  `GGML_BACKEND_SHARED` defined** (`ggml-backend-impl.h:267-293`) [SRC]; a different `api_version` is rejected
  (`ggml-backend-reg.cpp:246`) [SRC]. Both local llama.cpp builds are shared (`BUILD_SHARED_LIBS=ON`,
  `GGML_BACKEND_DL=OFF`), so a backend linked against the same `libggml-base.so` shares one copy of ggml [SRC: the
  build caches]; a static llama.cpp build would not work.
- **Correctness tool.** `test-backend-ops -b <device> -o MUL_MAT` compares against the CPU; the MUL_MAT NMSE limit is
  5e-4 (`test-backend-ops.cpp:5482-5484`) [SRC]. It allocates src0 **without** the WEIGHTS usage (only `test_bin_bcast`
  uses the weight context, line 3582) [SRC], so it never exercises a weight cache; modes `test`, `perf`, `support`.

## 3. Design: D0, weights stay where llama.cpp puts them, packed copies in the backend

Chosen for steps 1-2 (alternatives in section 3.3).

### 3.1 Mechanism

- One ACCEL device `gen9`. Default buffer type = `ggml_backend_cpu_buffer_type()`, `supports_buft` = host buffers,
  device memory reported as 0, as ggml-blas. llama.cpp then leaves every weight in its mmap buffer and our compute
  results in host memory: no copies in or out, any op we decline runs on the CPU in place.
- The backend context holds a **packed-weight cache**: key = (`src0->data`, `ggml_nbytes`, type, ne0, ne1); value = a
  packed copy in GPU memory allocated by gen9blas (`gen9blas_packed`, section 4).
- A MUL_MAT goes through the cache only if src0 is a **weight**: `src0->buffer` usage == WEIGHTS, `src0->op == NONE`,
  `src0->view_src == NULL`, no INPUT / PARAM flag, 2-D, contiguous. Any other src0 (test-backend-ops, computed
  matrices) is packed per call and not cached (today's BLAS path, which test-backend-ops then covers).
- Packing happens on the first use of a weight (llama-bench warm-up pass, first prompt). **Invariant: the cache lives
  in the backend context and dies with it; the backend dies with the llama_context, which llama-bench frees before the
  model (`llama-bench.cpp:2532, 2537`) [SRC].** Keying by data pointer is safe only under this invariant; the cache
  must never become process-global without invalidation on buffer free. Two contexts on one model each hold their own
  packed copies (~6 GB F32 each).
- WEIGHTS usage does not by itself mean immutable: MoE cache banks are rewritten (`llama-moe-cache.cpp:616-619`), but
  only through MUL_MAT_ID, which we do not accept; KV-cache rotation matrices (`llama-kv-cache.cpp:411`) carry the
  Hadamard hint, which we decline [SRC]. LoRA is applied at graph level (`llama-graph.cpp:1707-1713`) and adapter
  buffers have no WEIGHTS usage (`llama-adapter.cpp:393`) [SRC], so LoRA MUL_MATs take the per-call path and work.
  Control vectors are ADDs. `--no-mmap` changes only where the originals live. A debug option
  (`GEN9_CHECK_WEIGHTS=1`) re-checksums cached weights each graph to detect violations.
- The original pages are pinned (userptr) only while packing. llama.cpp maps the model `PROT_READ, MAP_SHARED`
  (`llama-mmap.cpp:715-725`) and kcl passes `I915_USERPTR_READ_ONLY` for inputs (`kcl.c:157`) [SRC]; the
  LD_PRELOAD runs already read these weights through userptr with llama-bench's default mmap load [MEAS], so this works.
  Two library changes are needed: (a) a kcl call to release a userptr object after packing (today objects stay in the
  LRU, pinned, up to `KCL_USERPTR_MB`); (b) since userptr resolution is lazy and a failure only appears at kernel
  argument / execbuf time (`kcl.c:519`), and `GEN9BLAS_NO_USERPTR` is read once (`gen9blas.c:278`), `pack_b` needs a
  per-call "copy through the CPU" flag and retries with it on failure.

### 3.2 Memory

Packed copies are about the weight size (padding to 16). The originals stay in the page cache while something on the
CPU still reads them: in step 1 token generation runs on the CPU, so both are hot (F32 1.5B: 6.2 + 6.6 GB on a 15 GB
machine: tight; F16: 3.1 + 3.3 GB). From step 2, when the GPU handles batch 1 as well, the CPU no longer touches the
weights and the kernel can drop those pages. Step 1 measures F16 first, F32 as a check.

### 3.3 Alternatives considered

- **D1, own non-host buffer with the packed copy only.** One copy, but: results of our MUL_MATs land in non-host memory
  and are copied out (1.32 GB per pp512 pass); src1 in CPU buffers is copied in unless `supports_buft` accepts host
  (0.77 GB per pass); any MUL_MAT we decline makes the scheduler copy the whole weight to the CPU through `get_tensor`
  on every graph (pass 5, `ggml-backend.cpp:1790-1815`) [SRC], e.g. 6.2 GB of unpacking per token. Also, with its own
  buffer type the device receives every op-NONE weight (`llama-model-loader.cpp:1017-1019`) [SRC]. Rejected.
- **D2, own host buffer with original + packed (ggml-zdnn pattern).** Robust, but two anonymous copies (F32 1.5B:
  12.4 GB) unless the device also implements `buffer_from_host_ptr` so llama.cpp wraps the mmap
  (`llama-model.cpp:2030-2054`) [SRC]; then it is D0 with more interface code. Rejected for now.
- **IGPU device type instead of ACCEL.** Needs `-ngl`, and llama.cpp keeps only the first IGPU device
  (`llama.cpp:260-266`) [SRC]. No gain for this design.

## 4. What gen9blas needs to expose

The library already packs once and launches many times internally (`pack_dev`, `launch` with row bands below
`GEN9BLAS_MAX_LAUNCH_MS` = 100 ms, `lib/gen9blas.c:404-486`) [SRC]; `roadmap.md` item 7 already notes that packed
weights need an explicit API. Proposed (names provisional):

```c
typedef struct gen9blas_packed gen9blas_packed;   // opaque: GPU allocation, type, N, K, Np, Kp
gen9blas_packed *gen9blas_pack_b(int type, int tB, int K, int N, const void *B, int ldb);  // op(B) is K x N; banded
int  gen9blas_gemm_pb(int type, int tA, int M, double alpha, const void *A, int lda,
                      const gen9blas_packed *Bp, double beta, void *C, int ldc);           // C = alpha op(A) Bp + beta C
int  gen9blas_gemv_pb(/* step 2: M <= 16 on the packed layout */);
void gen9blas_packed_free(gen9blas_packed *);
```

ggml MUL_MAT computes `dst[M x N] = src1[M x K] . src0^T`, src0 stored N x K row-major (`ne00 = K`): the same call
ggml-blas makes (RowMajor, NoTrans, Trans; `ggml-blas.cpp:142`) [SRC], so `pack_b` runs with `tB = 1` once per
weight. `gemm_pb` keeps activation packing (0.14 s per pass), 100 ms row bands, direct write into C when M and N are
tile multiples, userptr for the activations and C.

**Packing must be banded too (prerequisite of step 1).** `pack_dev` issues one launch (`lib/gen9blas.c:404-413`)
[SRC]. The measured packing rate is ~4 GB/s (1.26 s for ~5.1 GB of weights per pass) [MEAS], so packing
`output.weight` (1536 x 151936 F32, 0.93 GB) would be one ~230 ms launch, 2.3x the 100 ms policy [HYP from the rate].
`pack_b` splits the work into panel ranges of at most ~400 MB (~100 ms). Offsets inside kernels must be 64-bit for
weights above 2 GB (check `pkf_b_t` / `pk_b16_t` and `launch1` arguments).

**F16 weights.** ggml-cpu multiplies F16 weights with F32 activations and accumulates in F32 (`vec_dot_f16_f32`,
`ggml-cpu.c:225-229`) [SRC]. gen9blas's accurate FP16 GEMM (`GEN9BLAS_HS`: FP16 operands, products accumulated in
chunks into FP32, `lib/gen9blas.h:13`) [SRC] needs: activation packing that converts F32 -> F16 (new `pk_a3` variant)
and a result written as F32 (new epilogue / direct write). Precision: F16 rounding of activations gives NMSE far below
the 5e-4 limit [HYP]; the real risk is **overflow**, and not only of single activations: HS sums
`GEN9BLAS_KC` = 32 products in FP16 before adding them into FP32 (`lib/gen9blas.h:13-15`) [SRC], so a chunk sum above
65504 becomes inf where ggml-cpu stays in F32. llama.cpp's own F16 path leaves "x32 headroom for the F16 sums of 32
values" (`llama-graph.cpp:1736-1741`) [SRC]. Rules: (1) no saturation (it would hide the error); (2) before any GPU
work, a small `cb_eval` program (the stock `eval-callback` example prints tensors but not these maxima) logs, for every
MUL_MAT over the perplexity text, max |src1| and max |src1| x max |w| x 32 as a bound on the chunk sum; (3) at run time
the F32 epilogue / direct write reduces a non-finite flag, and a call with a non-finite result is rerun on the F32 path.

## 5. Kernels: what exists, what is new

| Need | Status |
|---|---|
| FP32 GEMM on packed B (`sgemm_v`) | exists; 365 GFLOPS on these shapes [MEAS] |
| FP16 accurate GEMM on packed B (`hgemm_kc_v`) | exists; whole call incl. packing 510-606 GFLOPS at n = 1024-4096 [MEAS, `results/gemm-sizes-2026-11-12.txt`] |
| pack B transposed, F32 / F16 (`pkf_b_t`, `pk_b16_t`) | exist; **need banding** (section 4) |
| pack A F32 -> F16 with overflow check | **new** (variant of `pk_a3`) |
| F16 GEMM writing F32 C | **new** epilogue / direct write |
| GEMV / skinny (M <= 16) on the packed layout | **new** (step 2). Layout: panels of 16 columns, for each k the 16 values contiguous (`PKF_IDX_B`, `bench/sgemm.cl:133`) [SRC]. [HYP] one subgroup per panel, lane = output column, Kp x 64 B streamed; N = 256 / 1536 give only 16 / 96 panels for 168 hardware threads, so K is split across subgroups and reduced |

## 6. Model and predictions

### 6.1 Prompt processing (pp512, llama-bench, last-token logits only)

Per pass of the measured model (1299 GFLOP of weight GEMM). Pinning: with weights cached the measured execbuf floor is
0.35-0.40 ms per call (`KCL_USERPTR_MB=8192` runs) [MEAS], x 193 = ~0.07 s.

| Version | GEMM | B packing | execbuf | A packing | CPU ops | Pass | pp512 [HYP] |
|---|---|---|---|---|---|---|---|
| today, BLAS path [MEAS, session B] | 3.56 | 1.26 | 0.20 | 0.14 | 0.85 | 6.08 s | 81-84 t/s |
| D0, F32 | 3.56 | 0 | ~0.07 | 0.14 | 0.85 | ~4.6 s | **105-115 t/s** |
| D0, F16 (HS at 550-640 GFLOPS [HYP: 1.5-1.75x the FP32 kernel, as in the whole-call ratio]) | 2.0-2.4 | 0 | ~0.07 | ~0.15 | 0.85 | 3.1-3.5 s | **145-165 t/s** |

The first pass of a run also packs (~1.3 s F32 / ~0.7 s F16 once); llama-bench's warm-up absorbs it. In llama-bench
the head runs with M = 1 (declined in step 1), so step 1 packs ~5.2 GB F32, not 6.2.

### 6.2 Token generation (batch 1)

Memory-bound GPU time = weight bytes per token / 20-24 GB/s (GPU RAM bandwidth in practice, `performance-model.md`)
[MEAS for the bandwidth]. The GGUF has a separate `output.weight` (1536 x 151936; `llama-quantize` log) [MEAS], so the
head runs on the GPU too: 6.2 GB F32 / 3.1 GB F16 per token. Fixed cost: one token is 28 x 7 + 1 = 197 MUL_MATs. If
each is its own submission at the measured 0.35-1.2 ms of execbuf CPU, that adds 0.07-0.24 s per token [MEAS for the
per-call cost], comparable to the GPU time. ggml groups consecutive nodes of one backend into one split, so
submitting **one batch buffer per split** saves submissions only where MUL_MATs are adjacent. In Qwen2 the Q, K and V
projections are each followed by a bias ADD (a CPU op) (`llama-graph.cpp:1920-1935`) and the scheduler does not
reorder nodes; gate and up are adjacent (both feed `ggml_swiglu_split`, `llama-graph.cpp:2083`) [SRC]. So per layer
there are 6 groups (Q; K; V; O; gate+up; down): 28 x 6 + 1 = 169 submissions per token instead of 197
(0.06-0.20 s). Every launch in a split is a short GEMV, so the preemption limit is not at risk. Not in the model:
CPU ops per token (small at batch 1), wake-up latency of ~170 CPU/GPU alternations per token, and the CPU threads
spinning meanwhile under the shared 15 W budget; step 2 measures a CPU/GPU round trip first.

| | GPU weights / token | GPU time | submissions | tg [HYP] | today |
|---|---|---|---|---|---|
| F32, one submission per MUL_MAT | 6.2 GB | 0.26-0.31 s | 0.07-0.24 s | 1.8-3.0 t/s | 2.28 (CPU), 3.87 (Vulkan) |
| F32, one per split | 6.2 GB | 0.26-0.31 s | 0.06-0.20 s | **2.0-3.1 t/s** | |
| F16, one per split | 3.1 GB | 0.13-0.16 s | 0.06-0.20 s | **2.8-5.3 t/s** | to measure (CPU, Vulkan) |

So token generation on the GPU pays off only with per-split submission and mostly for F16 and smaller types; F32 may
not beat the CPU and will not reach Vulkan's 3.87 t/s. This is the main uncertainty of step 2.

**Measured 2026-11-13 (F16, before writing step 2)** [MEAS]: one token replayed as its 197 FP16 GEMVs with the existing
`skinny_k1` on USM (`bench/token_gemv.py`, `bench/token_submit.py`; `results/token-gemv-2026-11-13-quiet.txt`,
`results/token-submit-2026-11-13*.txt`). Activations are FP16 here (F32 in the backend) and the CPU ops are left out.

| waits for the GPU per token | ms / token | t/s |
|---|---|---|
| 197 (one per GEMV) | 150-193 (machine state: 170-390 us per wait) | 5.2-6.7 |
| 169 (one per split, this section's plan) | 176-180 | 5.6-5.7 |
| 29 (one per layer) | 140-145 | 6.9-7.2 |
| 1 (one per token) | 115-120 | 8.3-8.7 |
| summed kernel time | 114-120 (26-27 GB/s of weights) | |
| llama.cpp CPU tg F16 (`llama-cpp-openvino-2026-11-13.txt`) | 145 | 6.9 |

Consequences: (1) the GEMV kernel is at the RAM ceiling (~27 GB/s), so a new packed GEMV (`gemv_pb`) can save at most
~2-5 ms per token and is dropped as a goal; the M = 1 path stays `skinny_k1`. (2) One submission per split saves
about 5 %: the bias ADDs after Q, K and V keep the splits almost one per MUL_MAT. (3) Beating the CPU needs the token
to run with few waits, i.e. the CPU ops between MUL_MATs (bias ADD, RMSNorm, RoPE, attention, SwiGLU) on the GPU;
one wait per layer only ties with the CPU, one per token gives about +25 %. Step 2 is redefined accordingly. The OpenVINO
GEMV kernel (`fully_connected_gpu_gemv.cl`, master 56346f1a) handles only INT4/UINT4 weights, so it is a reference
for step 3 (quantized types), not for F16; its weight layout `os_iyx_osv16` matches our packed B.

### 6.3 Falsification criteria (set now)

- Step 1: D0/F16 pp512 below 130 t/s, or D0/F32 below 100 t/s, in alternating runs against the BLAS path; perplexity
  different from ggml-cpu by more than 0.1 % (F32) / 0.5 % (F16); any GPU reset.
- Step 2: packed GEMV below 18 GB/s of weights in a microbenchmark; tg F16 not above the CPU's tg F16.

## 7. supports_op (step 1)

Accepted without work: `NONE`, `RESHAPE`, `VIEW`, `PERMUTE`, `TRANSPOSE`, as ggml-blas does
(`ggml-blas.cpp:404-409`) [SRC]. Two reasons: test-backend-ops asks `supports_op` for **every** tensor of a test
graph, leaves included, and reports the case unsupported on any false (`test-backend-ops.cpp:1482-1501`) [SRC], so a
MUL_MAT-only predicate would make every case "not supported"; and view ops do not break splits
(`ggml-backend.cpp:1733`), so they land inside our splits anyway. `graph_compute` skips them and every node without
`GGML_TENSOR_FLAG_COMPUTE` (`ggml-blas.cpp:232`) [SRC].

MUL_MAT accepted when all hold, otherwise declined (the CPU runs it in place):
- src0 type F32 or F16; 2-D (`ne2 = ne3 = 1`); contiguous; ne00 and ne01 fit in `int`;
- src1 type F32; contiguous; 2-D (`ne2 = ne3 = 1`) (true for all seven Qwen2 MUL_MATs during pp [SRC]);
- dst F32, contiguous;
- `ne1 >= 32` (batch); `ne0 >= 32`; `ne10 >= 32` (as ggml-blas);
- not `GGML_HINT_SRC0_IS_HADAMARD` (`ggml-blas.cpp:424-427`) [SRC];
- for F16 src0: neither `ggml_prec_get_acc(op)` (op_params 0) nor `ggml_prec_get_src(op, 1)` (op_params 3) asks for
  F32 or BF16 (`ggml.c:3365-3392`; zdnn declines these, `ggml-zdnn.cpp:122`) [SRC];
- weight-ness (section 3.1) decides cache vs per-call packing, not acceptance.

`init_backend` must not fail: llama.cpp throws if an ACCEL device returns NULL (`llama-context.cpp:491-493`) [SRC].
GPU initialisation is therefore lazy (first `graph_compute`), and a backend whose GPU init fails declines everything.

`test-backend-ops support -b gen9 -o MUL_MAT` lists which generated cases this accepts; the step-1 report states the
number of accepted cases next to "all pass", so that a pass is not vacuous (the generated cases include N = 100, 31,
K = 2051, broadcast 4-D, views [SRC: test-backend-ops case list]).

## 8. Plan, each step ending in a measurement

0. **Skeleton** (no GPU work). (a) A third llama.cpp 2cbe0ec build, **CPU only** (`GGML_BLAS=OFF`,
   `GGML_VULKAN=OFF`, shared libs, tests and examples on). Neither existing build can host the backend: in `build-blas`
   the statically registered BLAS device comes first in the ACCEL loop (`llama-context.cpp:487-496`) and takes every
   M >= 32 MUL_MAT, so gen9 would silently get nothing; in the Vulkan build CPU-side buffers can become `Vulkan_Host`
   (`llama-context.cpp:557-564`, `llama-model.cpp:1251-1257`) [SRC], which userptr cannot pin (GEM mmaps give EFAULT,
   `lib/kcl.c:108`) [SRC]. `test-backend-ops` exists only in `build/bin` and `llama-eval-callback` in neither [MEAS].
   (b) `libggml-gen9.so` built against `ggml/include` plus `ggml/src/ggml-backend-impl.h` and the build's shared
   `libggml-base.so`, compiled with `-DGGML_BACKEND_DL -DGGML_BACKEND_SHARED`, exporting `ggml_backend_init`; one ACCEL
   device; supports_op = view ops only; lazy `init_backend`; default buffer type = CPU. Check: loaded through
   `GGML_BACKEND_PATH`, listed by `llama-bench --list-devices` and `test-backend-ops support`; llama-bench unchanged.
1. **Prerequisites, then D0 for M >= 32.** (a) banded `pack_b` (and a banded or direct-per-band epilogue: the
   perplexity head's epilogue is one 311 MB launch otherwise), 64-bit offsets, per-call copy fallback and userptr
   release in the library, tested with `make -C lib check` plus a pack of `output.weight` size; (b) the `cb_eval`
   range check (section 4) on the CPU build; (c) `pack_b` / `gemm_pb` / F16 conversion with the non-finite flag;
   (d) the backend: cache, supports_op from section 7, and a routing counter (MUL_MATs taken by gen9 / declined, by
   reason) printed at exit. Checks: `test-backend-ops -b gen9 -o MUL_MAT` (count of accepted cases); a dedicated test
   that loads a model and compares logits of one prompt against ggml-cpu (exercises the cache, which test-backend-ops
   does not); perplexity with the settings of the results file; llama-bench pp512 / tg128 alternating against
   ggml-cpu, BLAS + gen9blas (`build-blas`) and Vulkan (`build`), with the routing counter in the report.
2. **Packed GEMV / skinny for M <= 16, one submission per split**; token generation on the GPU. Checks: first a
   CPU/GPU round-trip microbenchmark (169 alternations per token), then GB/s of weights for the GEMV, then tg128
   alternating; RSS (the CPU should stop touching the original pages). The falsification numbers for step 2 are set
   after the round-trip measurement.
3. Later, separate decision: quantized types (Q8_0, Q4_0, Q4_K): new GEMM and GEMV kernels.

## 9. Risks and open questions

- **GPU resets.** kcl has no mid-thread preemption; every launch must stay well below 640 ms. GEMM uses 100 ms row
  bands; packing gets the same (section 4). `llama-perplexity` needs all 512 rows of the head, 239 GFLOP per pass
  (~0.65 s at 365 GFLOPS as one launch): it must go through the bands. Vulkan hung exactly there [MEAS].
- **Weights changing after packing.** Covered by the weight-ness rule, the context-lifetime invariant of the cache and
  by declaring training unsupported (LoRA works through the per-call path); the debug checksum option detects
  violations.
- **Memory.** Packed copies are one GPU allocation per weight (largest: the head, 0.93 GB F32). [HYP] allocations of
  this size work through `gen9blas_alloc` / kcl; `kcl_userptr` refuses regions above `KCL_USERPTR_MB` (default
  1024 MB, `lib/kcl.c:205, 381`) [SRC], which matters only for the one-off pinning of an original while packing.
- **Thread safety.** gen9blas keeps global state (work buffers, one queue); `graph_compute` must hold a mutex if two
  llama contexts ever run at once.
- **API churn.** The backend interface changes between releases (`api_version` 3 now). Pin the llama.cpp commit.
- **Where the code lives.** [Proposal] a separate repository for the backend; library API additions here. Name and
  licence (MIT, as llama.cpp and this repository) to decide.
- **Who runs F32/F16.** Few users; value comes with Q8_0 / Q4 (step 3). Steps 0-2 establish the mechanism, the speedup
  and the safety on the simplest types.

## 10. Changes after review of draft 1 (2026-11-12)

1. New design D0 (weights in place, packed cache in the backend), replacing D2 for step 1 and D1 as a later step.
2. Corrected: ggml-blas does not receive weights through the ACCEL buffer list (`llama-model.cpp:1239` skips the CPU
   buffer type); it gets its nodes through the scheduler.
3. Added the cache validity rules and the fact that test-backend-ops does not exercise a weight cache; added a
   model-level test.
4. D1 described completely (copies in, copies out, weight copies on decline).
5. Token generation re-modelled with per-submission cost; F32 tg may not beat the CPU; per-split submission added.
6. Packing launches must be banded (`output.weight` ~230 ms as one launch); 64-bit offsets.
7. F16: overflow check instead of a tolerance worry; HS throughput taken from a measured file; supports_op declines
   Hadamard and precision hints.
8. Step 0 build flags (`GGML_BACKEND_DL`, `GGML_BACKEND_SHARED`), shared-library build requirement.
9. Memory: one allocation per weight instead of one huge buffer.
10. Section 1 separates the two sessions; qwen2 output-row selection cited from source; pinning floor 0.07 s; the
    supports_op predicate and accepted-case count are written down.

## 11. Changes after review of draft 2 (2026-11-12)

1. supports_op accepts the view ops and `NONE` (test-backend-ops asks for every tensor of a test graph);
   `graph_compute` skips them and nodes without the COMPUTE flag.
2. F16 overflow: the bound is the 32-product FP16 chunk sum of HS, not |src1|; no saturation; a non-finite flag in the
   epilogue reruns the call in F32; precision hints checked through both `ggml_prec_get_acc` and `ggml_prec_get_src`.
3. Step 0 adds a CPU-only llama.cpp build (BLAS would take the work first; the Vulkan build can put CPU-side buffers in
   `Vulkan_Host`, which userptr cannot pin); a small `cb_eval` program instead of the stock eval-callback; a routing
   counter in the step-1 report; lazy `init_backend` that never returns NULL.
4. Token generation: 6 groups per Qwen2 layer (169 submissions per token); upper bounds 3.1 / 5.3 t/s; round-trip
   microbenchmark before step-2 criteria.
5. userptr on the read-only mmap is already exercised by the LD_PRELOAD runs; library needs a userptr release call and
   a per-call copy fallback in `pack_b`.
6. Cache invariant written down (lives and dies with the context); LoRA works through the per-call path; MoE banks and
   KV rotation matrices excluded by MUL_MAT_ID / the Hadamard hint; two contexts duplicate the packed copies.
7. Step 1 packs ~5.2 GB F32 in llama-bench (head at M = 1 declined).
8. Banded epilogue for the perplexity head; `int` range check on ne; section reference fixed.

## 12. Status after implementation (2026-11-13)

Results files: `results/ggml-backend-step0-2026-11-12.txt`, `-step1-f32-`, `-step1-f16-`, `-ffn-fusion-`,
`-q1-dequant-2026-11-13.txt`, `results/llama-cpp-openvino-2026-11-13.txt`. Qwen2.5-1.5B-Instruct, UHD 620, pp512 t/s,
alternating runs in one session:

| Weights | GEN9 | ggml-cpu | OpenVINO GPU | GEN9 perplexity vs reference |
|---|---|---|---|---|
| F32 | 107-109 | 46-49 | (not supported) | +0.001 % |
| F16 | 177-184 | 47-52 | 182-187 | +0.017 % |
| Q8_0 | 183-186 | 51-53 | 139-140 | -0.18 % vs CPU Q8_0 |
| Q4_K_M | 183 | 77-82 | 45 | -0.012 % vs CPU on the dequantized weights |

Other architectures, Q4_K_M (`results/ggml-backend-architectures-2026-11-13.txt`): Llama-3.2-1B 248-253 t/s vs CPU
102-104 and OpenVINO 89 (2.4x CPU); gemma-2-2b 110-113 vs 49 and 42 (2.3x). Every weight MUL_MAT ran on GEN9 in both;
perplexity -0.07 % / -0.46 % vs ggml-cpu (which quantizes activations to Q8_K).

What was built: design D0 (section 3), weights packed once into a per-backend cache; F16 and every quantized type with
`to_float` go through the accurate FP16 GEMM (quantized types dequantized to FP16 at first use, step Q1); an overflow
flag in the epilogue recomputes a call in F32; a device buffer type `GEN9_Host` (host memory, standard layout) keeps the
weights out of ggml-cpu's REPACK buffer, which GEN9 cannot read. Token generation stays on the CPU (batch < 32).

What was learned:
- Loading: common/ tools ignore `GGML_BACKEND_PATH` in a shared non-DL build; `-ot '^gen9_load_backends_only$=CPU'`
  forces `ggml_backend_load_all()`. Always check the routing counter (`GEN9_INFO=1`) before trusting a number.
- Power: this laptop (PL1 25 W, 85-95 C under load) shares one budget between CPU and GPU. The fused FFN block (stage A,
  `GEN9_FFN=1`) removes GPU and CPU work but is not faster: the CPU clock falls from ~2500 to ~1750 MHz and CPU flash
  attention slows by 48 %. Moving more work to the GPU pays only where it cuts energy, not just time; energy (RAPL) has
  not been measured yet (needs root).
- Token generation: the GPU GEMV is at the RAM ceiling like the CPU (separate work: 197 F16 GEMVs = 117 ms; waits per
  token decide); OpenVINO, which runs everything on the GPU, reaches 7.3 t/s on F16 vs 6.9 on the CPU and 7.5 vs 23.5 on
  Q4_K_M. A quantized GPU GEMV was dropped (an external review); token generation stays on the CPU.
- Q1 first-use cost: 5.1-5.4 s for the 1.5B model (scalar `to_float` 3.2 s, packing 1.7 s); a GPU dequant kernel is the
  next lever, and with an LRU budget the way to models above ~4 B parameters (memory: packed FP16 = 2 B per weight).

Open: GPU dequant kernel; energy per token (RAPL); other architectures (Llama, Gemma: RoPE / GELU variants stay on the
CPU, MUL_MAT is generic); stage B (QKV) only if energy says so; separate repository and name.

### 12.x Update 2026-11-13 (day): GPU dequantization, models above the memory budget, energy

- Q8_0 / Q4_K / Q6_K are dequantized on the GPU straight into the packed layout (`gen9blas_pack_b_q`, kernel
  `gen9blas_dequant_b16`), bit-identical to ggml's `dequantize_row_*` + RTE FP16 (checked per weight on three models
  with `GEN9_DEQ_VERIFY=1`). First use for Qwen2.5-1.5B: Q4_K_M 5.6 -> 2.3 s, Q8_0 5.3 -> 1.9 s. Other types keep
  the CPU path (Q1). `results/ggml-backend-gpu-dequant-2026-11-13.txt`.
- Cache budget `GEN9_CACHE_MB` (default 40 % of RAM): weights cached in first-use order, the rest dequantized again
  on every use (cyclic access: keeping the first ones beats LRU). Qwen2.5-7B Q4_K_M (14 GB FP16): pp512 29.2-29.8 vs
  ggml-cpu 12.6-13.4 t/s (2.2-2.3x), perplexity -0.30 %, peak 9.7 GB. `results/ggml-backend-7b-budget-2026-11-13.txt`.
- Energy (RAPL): GEN9 ~115 vs ggml-cpu ~368 mJ per prompt token (3.2x less). FFN fusion: 113 vs 127 mJ/token
  (-10.7 %) at -1-3 % speed: slower but cheaper, because the lower CPU clock makes the CPU work cheaper.
  `results/ggml-backend-energy-2026-11-13.txt`.
- Lesson: a kernel that passed the unit test at N <= 160 was wrong from N ~ 1024 on real models (Q4_K scale decode in
  private arrays; cause not found, scalar rewrite is exact). Unit tests now include N = 4096; `GEN9_DEQ_VERIFY=1`
  checks a real model end to end.
- Flash attention on the GPU (2026-11-13 evening, on by default, `GEN9_ATTN=0` turns it off, [attention-gpu.md](attention-gpu.md) section
  11): FLASH_ATTN_EXT with F16 K/V and the single-sequence causal mask runs the hand-scheduled kernel through kcl on
  userptr; anything else stays on ggml-cpu. HD 530: pp512 at depth 3584 83.8 -> 131.0 t/s (Qwen2.5-1.5B, -36 % time)
  and 104.7 -> 182.5 (Llama-3.2-1B, -43 %), d0 +11 % / +18 %; perplexity -0.007 % / -0.008 % vs GEN9_ATTN=0.
  `results/attn-integration-hd530-2026-11-13.txt`.
- Per-layer elementwise ops on the GPU (2026-11-13 night, `GEN9_EWOPS=0` / `GEN9_QKV=0` turn them off): RMS_NORM with its
  weight MUL, ADD, MUL and ROPE only enqueue (no own wait; the ggml buffers of a split are registered as kcl spans so each
  byte has one GPU address in a batch), and up to 2 of the 3 Q/K/V MUL_MATs share one packed input and one wait. pp512
  +2.5 % on gp, perplexity +/-0.01 %. `results/ggml-backend-gpu-ops-2026-11-13.txt`.
- **Measurement rule (from 2026-11-13 night): every llama.cpp measurement with GEN9 uses `-ngl 0` and states the `-ngl`
  used in its results file**, because with the default `-ngl` (full offload) llama.cpp pins every layer's norm to the
  CPU (`src/llama-context.cpp:3257-3265`), and GEN9 (an ACCEL device) is affected by `-ngl` in no other way; older results
  are not recomputed, but a new-vs-old comparison needs a fresh baseline with the same `-ngl`.

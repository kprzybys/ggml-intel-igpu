# Lessons from profiling the vendor stack (OpenVINO on Gen9) and how they generalize

Source: whisper large-v3-turbo on the UHD 620 laptop, 2026-11-13, `results/whisper-large-v3-turbo-gp-2026-11-13.txt`
(end-to-end comparison and the OpenVINO GPU plugin profile). Tags: [MEAS] measured, [EST] estimate to be checked.

## 1. What the profile showed

| Piece | OpenVINO GPU plugin | GEN9 (ggml backend) | Reading |
|---|---|---|---|
| Encoder attention | `ocl::sdpa::opt` ~230 GFLOPS [MEAS] | asm kernel 334 GFLOPS [MEAS] | ours ~45 % faster |
| Encoder FullyConnected | `fully_connected_gpu_bf_tiled` ~4.5 s per window (~420 GFLOPS), bias / GELU / residual fused [MEAS] | GEMM ~2.8 s + packing/epilogue ~0.57 s per window; bias, GELU, adds on the CPU (~0.9 s/window with conv and waits) [EST from kernel sums, results/whisper-profile-gp-2026-11-13.txt] | ours faster per GEMM; the vendor saves the CPU ops by fusion |
| Input convolution | `convolution_gpu_ref` (reference kernel), 10 % of the encoder [MEAS] | on the CPU | not tuned for Gen9 by the vendor |
| Encoder per 30 s window | ~5.6 s [MEAS, PERF_COUNT on] | 5.39 s [MEAS] | a draw |
| Decoder per token | GPU ~44 ms [MEAS, PERF_COUNT on] | CPU 23 ms [MEAS] | keep the decoder on the CPU |

End to end OpenVINO GenAI still wins (240-253 s vs 258-263 s for 645.6 s of audio) although our encoder window is
as fast and our decoder token twice as fast. Correction (profile of a whole whisper-cli run,
results/whisper-profile-gp-2026-11-13.txt): nothing is unaccounted; whisper.cpp ran the encoder **33 times, not 22**
(its seek moves to the last segment end, so windows overlap): encode 177 s = 69 %, decode 68 s (2870 tokens) = 26 %,
prompt 3 %, sampling 1 %, mel 0.3 %. Checked [MEAS, bpftrace on ov::InferRequest]: GenAI encodes **28** windows of the
same file (its seek overlaps less), 2621 tokens at ~31 ms per decoder step. GenAI's whole edge is ~5 fewer encoder
windows (~27 s); our CPU decoder wins back ~10 s. The largest lever is in the application (how many windows are
encoded), not in a kernel.
The OpenVINO profile is one run with PERF_COUNT on and random features: per-op figures are +-10-15 %.

## 2. Rules that carry over

**a) The vendor profile is a free per-op target on the same chip.** `PERF_COUNT` gives time and kernel name per
primitive. A faster vendor op tells that and by how much ours can improve (here it showed the opposite: our GEMM and
attention are faster; the vendor's edge is fusion and the pipeline). Compare whole runs too: a per-op comparison
missed that the two applications encode a different number of windows. `*_ref` kernels
near the top of the list mark where the vendor has dropped Gen9 (cheap wins for us). Other signs of that on Gen9: the
INT8 (NNCF) encoder hangs the GPU, INT4 does not build (earlier project), long kernels are not preemptible.
Usable for any model OpenVINO runs.

**b) "GEMM + epilogue" is the common transformer pattern.** OpenVINO folded 2461 IR ops into 457 primitives, nearly all
of the form `y = act(x W + b) + residual`. One parametric GEMM epilogue (bias; GELU or SiLU / SwiGLU gate; residual)
covers: whisper / wav2vec2 / parakeet (bias + GELU), BERT-type embedding and rerank models (bias + GELU + residual),
VLM image encoders such as CLIP / SigLIP (bias + GELU), Qwen-type LLMs (QKV bias; SwiGLU exists).
Caveat from llama.cpp [MEAS]: the fused FFN was not faster there (CPU and GPU share 25 W). The gain of a fusion is less
CPU work and fewer synchronisation points; bound it before building it:
splits per pass x wait cost (170-390 us) + CPU op time.

**c) Prefill on the GPU, token generation on the CPU (on Gen9 with shared DDR4).** Third independent confirmation:
OpenVINO's GPU decoder (44 ms/token) loses to the CPU (23 ms), as the token GEMV did in llama.cpp. GEMV is at the RAM
bandwidth ceiling either way, so the GPU only adds overhead. Applies to every autoregressive decoder (LLM, whisper's
decoder, TTS): batched work (encoder, prompt) to the GPU, one-token steps stay on the CPU.

**d) The backend serves any ggml application unchanged.** whisper.cpp needed no change; the same holds for llama.cpp
embeddings, `mtmd` image encoders and parakeet. A new attention shape now costs one Makefile target (the hex
immediate fix in `asm/attn_gen.py` removed the power-of-two row assumption).

**e) Robustness is an advantage of its own.** OpenVINO hung the GPU at i915's default 640 ms preemption timeout (the
heartbeat cannot preempt its long kernels); kcl splits launches below 100 ms. On machines with a desktop that is the
difference between working and resetting the GPU.

**f) Measuring.** Compare per stage (encoder per window, decoder per token), not only end to end; check every variant
ran as intended (backend counters, one process per run: Python 3.14 `forkserver` re-imports a script without a
`__main__` guard and ran a second transcription in parallel); use the application's default settings (whisper.cpp's
`-nf` hid a decoder loop that the default temperature fallback catches, and made the GPU variants look 10 points worse).

## 3. Next steps, in order

1. Done (separately, 493e500, and the window count above). Next: can whisper.cpp encode fewer windows (33 now, GenAI 28,
   21.5 nominal) without losing WER (seek / timestamp / max-len settings)? The encoder is 69 % of the run.
2. Done: our GEMM is ~2.8 s per window (+0.57 s packing/epilogue) vs OpenVINO's fused FullyConnected ~4.5 s.
   Decoder on the GPU [EST]: F16 weights, ~1.3x on the decoder = ~6 % of the run; measure before building.
3. Parametric GEMM epilogue (bias, GELU, residual): one kernel for whisper, embeddings and image encoders; measure with
   clocks and energy.
4. Input convolution on the GPU through the existing GEMM with F16 activations (small change; the vendor uses a
   reference kernel there).
5. Same measurement for an embedding model (e.g. bge-m3 in llama.cpp) and a VLM image encoder, to test 2d cheaply.

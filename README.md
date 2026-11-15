# ggml-intel-igpu

GEN9: a [ggml](https://github.com/ggml-org/ggml) backend for Intel Gen9 integrated GPUs (UHD 620 / HD 530), built
on [gen9-gpu-blas](https://github.com/kprzybys/gen9-gpu-blas). It works without Intel's OpenCL runtime: the library
drives the GPU through i915. llama.cpp and whisper.cpp load it as a dynamic backend (`GGML_BACKEND_PATH`). The
prompt-processing matrix products run on the GPU with weights packed once (FP16, or dequantized on the GPU from
Q8_0 / Q4_K / Q6_K), as do flash attention for the supported head shapes and the per-layer elementwise ops. Token
generation stays on the CPU.

- Design and evidence: [docs/ggml-backend-design.md](docs/ggml-backend-design.md).
- Measurements: `results/` (e.g. `compare-real-models-2026-11-13.txt`, `ggml-backend-energy-2026-11-13.txt`).
- Lessons from vendor profiles: [docs/vendor-profile-lessons.md](docs/vendor-profile-lessons.md).

## Build

Requirements:
- gen9-gpu-blas 0.2, installed (`cmake --install`, see its README);
- a shared, CPU-only ggml build of llama.cpp at commit 2cbe0ec: `cmake -B build-cpu -DGGML_BLAS=OFF
  -DGGML_VULKAN=OFF -DBUILD_SHARED_LIBS=ON -DLLAMA_BUILD_TESTS=ON`.

```sh
cmake -S . -B build -G Ninja -DCMAKE_PREFIX_PATH=<gen9blas prefix> -DGGML_SOURCE_DIR=~/src/llama.cpp
cmake --build build
GGML_BACKEND_PATH=$PWD/build/ggml-backend/libggml-gen9.so ~/src/llama.cpp/build-cpu/bin/llama-bench -m model.gguf -ngl 0
```

whisper.cpp's ggml (no `ggml_rope_get_truncate`): `-DGEN9_GGML_NO_ROPE_TRUNCATE=ON` with `GGML_SOURCE_DIR` set to the
whisper.cpp tree. `build/ggml-backend/test_vision`, `test_dec` and `test_attn` compare GPU results with ggml-cpu.

License: MIT (LICENSE).

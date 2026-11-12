# ggml backend GEN9 (docs/ggml-backend-design.md). Needs a shared, CPU-only llama.cpp build at the pinned commit 2cbe0ec:
#   cmake -B build-cpu -DGGML_BLAS=OFF -DGGML_VULKAN=OFF -DBUILD_SHARED_LIBS=ON -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=ON
LLAMA ?= $(HOME)/src/llama.cpp
LLAMA_BUILD ?= $(LLAMA)/build-cpu
CXXFLAGS = -std=c++17 -O2 -Wall -Wno-unused-function -fPIC -DGGML_BACKEND_DL -DGGML_BACKEND_SHARED -DGGML_BACKEND_BUILD \
           -I$(LLAMA)/ggml/include -I$(LLAMA)/ggml/src -I../lib -DGEN9_BENCH_DIR='"$(abspath ../bench)"'

libggml-gen9.so: ggml-gen9.cpp ../lib/gen9blas.h ../lib/libgen9blas.so
	g++ $(CXXFLAGS) -shared -o $@ $< -L$(LLAMA_BUILD)/bin -lggml-base -lggml-cpu -L../lib -lgen9blas \
	    -Wl,-rpath,$(LLAMA_BUILD)/bin -Wl,-rpath,$(abspath ../lib)

clean:
	rm -f libggml-gen9.so
.PHONY: clean

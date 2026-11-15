# ggml-intel-igpu: the GEN9 ggml backend against an installed gen9blas (find_package) and a ggml build.
#   cmake -S . -B build -DGGML_SOURCE_DIR=~/src/llama.cpp -DCMAKE_PREFIX_PATH=<gen9blas prefix>
cmake_minimum_required(VERSION 3.21)
project(ggml-intel-igpu VERSION 0.2.0 LANGUAGES CXX)
add_subdirectory(ggml-backend)

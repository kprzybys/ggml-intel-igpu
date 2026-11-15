# ggml backend GEN9 (docs/ggml-backend-design.md): libggml-gen9.so, loaded by llama.cpp / whisper.cpp through
# GGML_BACKEND_PATH (dynamic backend). Needs a shared, CPU-only ggml build (llama.cpp at the pinned commit 2cbe0ec):
#   cmake -B build-cpu -DGGML_BLAS=OFF -DGGML_VULKAN=OFF -DBUILD_SHARED_LIBS=ON -DLLAMA_BUILD_TESTS=ON
# and libgen9blas (find_package(gen9blas) or the gen9blas target of the top-level build).
#   cmake -S ggml-backend -B build -DGGML_SOURCE_DIR=~/src/llama.cpp [-DGGML_LIB_DIR=~/src/llama.cpp/build-cpu/bin]
#   whisper.cpp's ggml (no ggml_rope_get_truncate): -DGEN9_GGML_NO_ROPE_TRUNCATE=ON
cmake_minimum_required(VERSION 3.21)
project(ggml-gen9 VERSION 0.2.0 LANGUAGES CXX)
include(GNUInstallDirs)

set(GGML_SOURCE_DIR "$ENV{HOME}/src/llama.cpp" CACHE PATH "llama.cpp or whisper.cpp source tree (ggml/include, ggml/src)")
set(GGML_LIB_DIR "${GGML_SOURCE_DIR}/build-cpu/bin" CACHE PATH "directory with libggml-base.so and libggml-cpu.so")
option(GEN9_GGML_NO_ROPE_TRUNCATE "ggml without ggml_rope_get_truncate (whisper.cpp)" OFF)
option(GEN9_BUILD_TESTS "test_vision / test_dec / test_attn (GPU vs ggml-cpu)" ON)

if(NOT TARGET gen9blas::gen9blas)
  find_package(gen9blas 0.2 REQUIRED)
endif()
foreach(h ggml-backend.h ggml-cpu.h)
  if(NOT EXISTS ${GGML_SOURCE_DIR}/ggml/include/${h})
    message(FATAL_ERROR "ggml-gen9: ${GGML_SOURCE_DIR}/ggml/include/${h} not found (set GGML_SOURCE_DIR)")
  endif()
endforeach()
find_library(GGML_BASE ggml-base PATHS ${GGML_LIB_DIR} NO_DEFAULT_PATH REQUIRED)
find_library(GGML_CPU ggml-cpu PATHS ${GGML_LIB_DIR} NO_DEFAULT_PATH REQUIRED)
find_library(GGML_MAIN ggml PATHS ${GGML_LIB_DIR} NO_DEFAULT_PATH REQUIRED)

set(CMAKE_CXX_STANDARD 17)
add_library(ggml-gen9 SHARED ggml-gen9.cpp)
target_compile_definitions(ggml-gen9 PRIVATE GGML_BACKEND_DL GGML_BACKEND_SHARED GGML_BACKEND_BUILD
  $<$<BOOL:${GEN9_GGML_NO_ROPE_TRUNCATE}>:GEN9_GGML_NO_ROPE_TRUNCATE>)
target_include_directories(ggml-gen9 PRIVATE ${GGML_SOURCE_DIR}/ggml/include ${GGML_SOURCE_DIR}/ggml/src)
target_compile_options(ggml-gen9 PRIVATE -Wall -Wno-unused-function)
target_link_libraries(ggml-gen9 PRIVATE gen9blas::gen9blas ${GGML_BASE} ${GGML_CPU})
# installed next to libgen9blas ($ORIGIN) and finding the ggml it was built against (GGML_LIB_DIR)
set_target_properties(ggml-gen9 PROPERTIES PREFIX lib INSTALL_RPATH "$ORIGIN;${GGML_LIB_DIR}")
install(TARGETS ggml-gen9 LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if(GEN9_BUILD_TESTS)
  foreach(t test_vision test_dec test_attn)
    add_executable(${t} ${t}.cpp)
    target_include_directories(${t} PRIVATE ${GGML_SOURCE_DIR}/ggml/include)
    target_compile_options(${t} PRIVATE -Wall)
    target_link_libraries(${t} PRIVATE ${GGML_MAIN} ${GGML_BASE} ${GGML_CPU})
    set_target_properties(${t} PROPERTIES BUILD_RPATH "${GGML_LIB_DIR}")
  endforeach()
endif()

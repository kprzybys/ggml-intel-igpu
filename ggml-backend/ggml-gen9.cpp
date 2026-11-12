// ggml backend "GEN9" on gen9blas kernels -- step 0 skeleton (docs/ggml-backend-design.md, section 8).
// One ACCEL device. Default buffer type = the CPU buffer type, as ggml-blas: weights stay in llama.cpp's own (mmap)
// buffers and results stay in host memory. Step 0 accepts only the no-op nodes (NONE and the view ops) and computes
// nothing; it exists to prove loading through GGML_BACKEND_PATH and the scheduler wiring, without touching the GPU.
//
// Build (against a shared, CPU-only llama.cpp build; see Makefile):
//   g++ -std=c++17 -O2 -fPIC -shared -DGGML_BACKEND_DL -DGGML_BACKEND_SHARED -DGGML_BACKEND_BUILD
//       -I$LLAMA/ggml/include -I$LLAMA/ggml/src ggml-gen9.cpp -L$LLAMA/build-cpu/bin -lggml-base -o libggml-gen9.so
// Use: GGML_BACKEND_PATH=$PWD/libggml-gen9.so llama-bench ...      GEN9_INFO=1 prints the routing counter at exit.

#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"   // struct ggml_cgraph

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// ---------------- routing counter (printed at exit with GEN9_INFO=1) ----------------

static std::atomic<long> n_graphs{0}, n_nodes_seen{0}, n_noop{0}, n_mul_mat_run{0};
static std::atomic<long> n_mm_asked{0}, n_mm_declined{0};

static void gen9_report(void) {
    fprintf(stderr, "gen9: graph_compute %ld calls, %ld nodes (%ld no-op, %ld MUL_MAT run); supports_op MUL_MAT asked %ld, declined %ld\n",
            n_graphs.load(), n_nodes_seen.load(), n_noop.load(), n_mul_mat_run.load(), n_mm_asked.load(), n_mm_declined.load());
}

// ---------------- backend (stream) ----------------

static const char * gen9_backend_get_name(ggml_backend_t) { return "GEN9"; }

static void gen9_backend_free(ggml_backend_t backend) { delete backend; }

static enum ggml_status gen9_graph_compute(ggml_backend_t, struct ggml_cgraph * cgraph) {
    n_graphs++;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];
        n_nodes_seen++;
        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) { n_noop++; continue; }
        switch (node->op) {
            case GGML_OP_NONE: case GGML_OP_RESHAPE: case GGML_OP_VIEW: case GGML_OP_PERMUTE: case GGML_OP_TRANSPOSE:
                n_noop++;
                break;
            default:
                GGML_ABORT("gen9: unsupported op %s reached graph_compute", ggml_op_desc(node));
        }
    }
    return GGML_STATUS_SUCCESS;
}

static struct ggml_backend_i gen9_backend_i = {
    /* .get_name            = */ gen9_backend_get_name,
    /* .free                = */ gen9_backend_free,
    /* .set_tensor_async    = */ NULL,
    /* .get_tensor_async    = */ NULL,
    /* .set_tensor_2d_async = */ NULL,
    /* .get_tensor_2d_async = */ NULL,
    /* .cpy_tensor_async    = */ NULL,
    /* .synchronize         = */ NULL,
    /* .graph_plan_create   = */ NULL,
    /* .graph_plan_free     = */ NULL,
    /* .graph_plan_update   = */ NULL,
    /* .graph_plan_compute  = */ NULL,
    /* .graph_compute       = */ gen9_graph_compute,
    /* .event_record        = */ NULL,
    /* .event_wait          = */ NULL,
    /* .graph_optimize      = */ NULL,
};

static ggml_guid_t gen9_guid(void) {
    static ggml_guid guid = { 0x67, 0x65, 0x6e, 0x39, 0x62, 0x6c, 0x61, 0x73, 0x2d, 0x75, 0x68, 0x64, 0x36, 0x32, 0x30, 0x01 };
    return &guid;
}

ggml_backend_reg_t ggml_backend_gen9_reg(void);

// Must never return NULL: llama.cpp throws for an ACCEL device that fails (llama-context.cpp:491-493).
// GPU initialisation will be lazy (first MUL_MAT), from step 1 on.
static ggml_backend_t gen9_device_init_backend(ggml_backend_dev_t, const char *) {
    return new ggml_backend {
        /* .guid    = */ gen9_guid(),
        /* .iface   = */ gen9_backend_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_gen9_reg(), 0),
        /* .context = */ nullptr,
    };
}

// ---------------- device ----------------

static const char * gen9_dev_get_name(ggml_backend_dev_t) { return "GEN9"; }
static const char * gen9_dev_get_description(ggml_backend_dev_t) { return "Intel Gen9 GPU via gen9blas (kcl, i915)"; }
static void gen9_dev_get_memory(ggml_backend_dev_t, size_t * free, size_t * total) { *free = 0; *total = 0; }   // as ggml-blas
static enum ggml_backend_dev_type gen9_dev_get_type(ggml_backend_dev_t) { return GGML_BACKEND_DEVICE_TYPE_ACCEL; }

static void gen9_dev_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name = gen9_dev_get_name(dev);
    props->description = gen9_dev_get_description(dev);
    gen9_dev_get_memory(dev, &props->memory_free, &props->memory_total);
    props->type = gen9_dev_get_type(dev);
    props->device_id = "0000:00:02.0";
    props->caps = {
        /* .async                = */ false,
        /* .host_buffer          = */ false,
        /* .buffer_from_host_ptr = */ true,
        /* .events               = */ false,
        /* .mmap_support         = */ true,
    };
}

static ggml_backend_buffer_type_t gen9_dev_get_buffer_type(ggml_backend_dev_t) { return ggml_backend_cpu_buffer_type(); }

static ggml_backend_buffer_t gen9_dev_buffer_from_host_ptr(ggml_backend_dev_t, void * ptr, size_t size, size_t) {
    return ggml_backend_cpu_buffer_from_ptr(ptr, size);
}

static bool gen9_dev_supports_op(ggml_backend_dev_t, const struct ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_NONE: case GGML_OP_RESHAPE: case GGML_OP_VIEW: case GGML_OP_PERMUTE: case GGML_OP_TRANSPOSE:
            return true;   // no-op nodes: test-backend-ops asks for every tensor of a test graph
        case GGML_OP_MUL_MAT:
            n_mm_asked++; n_mm_declined++;   // step 0: nothing computed yet
            return false;
        default:
            return false;
    }
}

static bool gen9_dev_supports_buft(ggml_backend_dev_t, ggml_backend_buffer_type_t buft) { return ggml_backend_buft_is_host(buft); }

static const struct ggml_backend_device_i gen9_device_i = {
    /* .get_name             = */ gen9_dev_get_name,
    /* .get_description      = */ gen9_dev_get_description,
    /* .get_memory           = */ gen9_dev_get_memory,
    /* .get_type             = */ gen9_dev_get_type,
    /* .get_props            = */ gen9_dev_get_props,
    /* .init_backend         = */ gen9_device_init_backend,
    /* .get_buffer_type      = */ gen9_dev_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ gen9_dev_buffer_from_host_ptr,
    /* .supports_op          = */ gen9_dev_supports_op,
    /* .supports_buft        = */ gen9_dev_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

// ---------------- registry ----------------

static const char * gen9_reg_get_name(ggml_backend_reg_t) { return "GEN9"; }
static size_t gen9_reg_get_device_count(ggml_backend_reg_t) { return 1; }

static ggml_backend_dev_t gen9_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);
    static ggml_backend_device dev = { /* .iface = */ gen9_device_i, /* .reg = */ reg, /* .context = */ nullptr };
    return &dev;
}

static const struct ggml_backend_reg_i gen9_reg_i = {
    /* .get_name         = */ gen9_reg_get_name,
    /* .get_device_count = */ gen9_reg_get_device_count,
    /* .get_device       = */ gen9_reg_get_device,
    /* .get_proc_address = */ NULL,
};

ggml_backend_reg_t ggml_backend_gen9_reg(void) {
    static struct ggml_backend_reg reg = { /* .api_version = */ GGML_BACKEND_API_VERSION, /* .iface = */ gen9_reg_i, /* .context = */ NULL };
    static bool once = [] { if (getenv("GEN9_INFO")) atexit(gen9_report); return true; }();
    (void) once;
    return &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_gen9_reg)

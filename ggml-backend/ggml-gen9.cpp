// ggml backend "GEN9" on gen9blas kernels (docs/ggml-backend-design.md). Step 1: MUL_MAT with F32 weights.
// One ACCEL device. Default buffer type = the CPU buffer type, as ggml-blas: weights stay in llama.cpp's own (mmap)
// buffers and results stay in host memory (design D0). Weights (WEIGHTS-usage buffer, op NONE, not a view, not an
// input / parameter) are packed once into GPU memory (gen9blas_pack_b) and kept in a per-backend cache; any other src0
// goes through gen9blas_gemm per call. Nodes we decline run on the CPU in place.
//
// Build: see Makefile (shared, CPU-only llama.cpp build at commit 2cbe0ec; ../lib/libgen9blas.so).
// Use:   GGML_BACKEND_PATH=$PWD/libggml-gen9.so llama-bench ...      GEN9_INFO=1 prints the routing counter at exit.

#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"   // struct ggml_cgraph
#include "gen9blas.h"

#include <atomic>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <tuple>

#ifndef GEN9_BENCH_DIR
#define GEN9_BENCH_DIR ""
#endif

// ---------------- routing counter (printed at exit with GEN9_INFO=1) ----------------

static std::atomic<long> n_graphs{0}, n_noop{0}, n_mm_cached{0}, n_mm_packed{0}, n_mm_percall{0};
static std::atomic<long> n_mm_asked{0}, n_mm_declined_shape{0}, n_mm_declined_type{0}, n_mm_declined_batch{0},
                         n_mm_declined_hint{0}, n_mm_declined_gpu{0};

static void gen9_report(void) {
    fprintf(stderr, "gen9: graph_compute %ld; no-op nodes %ld; MUL_MAT run: cached weight %ld, packed now %ld, per call %ld\n",
            n_graphs.load(), n_noop.load(), n_mm_cached.load(), n_mm_packed.load(), n_mm_percall.load());
    fprintf(stderr, "gen9: supports_op MUL_MAT asked %ld, declined: shape/layout %ld, type %ld, batch<32 %ld, hint %ld, no GPU %ld\n",
            n_mm_asked.load(), n_mm_declined_shape.load(), n_mm_declined_type.load(), n_mm_declined_batch.load(),
            n_mm_declined_hint.load(), n_mm_declined_gpu.load());
}

// ---------------- GPU (lazy: first use) ----------------

static std::mutex gpu_mu;   // gen9blas keeps global state (work buffers, one queue)

static bool gpu_ok(void) {
    static int state = -1;   // -1 not tried, 0 failed, 1 ready
    std::lock_guard<std::mutex> lk(gpu_mu);
    if (state < 0) {
        state = gen9blas_init(GEN9_BENCH_DIR) == 0;
        if (!state) fprintf(stderr, "gen9: GPU init failed (%s); declining everything\n", gen9blas_error());
    }
    return state == 1;
}

// ---------------- backend (stream) ----------------

struct wkey {
    const void * data; size_t nbytes; int type; int64_t ne0, ne1;
    bool operator<(const wkey & o) const { return std::tie(data, nbytes, type, ne0, ne1) < std::tie(o.data, o.nbytes, o.type, o.ne0, o.ne1); }
};

// The cache lives and dies with the backend, which llama.cpp frees with the context, before the model
// (design section 3.1): a data pointer cannot be reused for another weight while an entry exists.
struct gen9_context {
    std::map<wkey, gen9blas_packed *> cache;
    ~gen9_context() { std::lock_guard<std::mutex> lk(gpu_mu); for (auto & kv : cache) gen9blas_packed_free(kv.second); }
};

static bool is_weight(const struct ggml_tensor * w) {
    return w->buffer && w->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS && w->op == GGML_OP_NONE &&
           w->view_src == nullptr && (w->flags & (GGML_TENSOR_FLAG_INPUT | GGML_TENSOR_FLAG_PARAM)) == 0;
}

static const char * gen9_backend_get_name(ggml_backend_t) { return "GEN9"; }

static void gen9_backend_free(ggml_backend_t backend) {
    delete (gen9_context *) backend->context;
    delete backend;
}

// dst[M x N] = src1[M x K] . src0^T with src0 stored N x K row-major: gemm(A = src1, B = src0 transposed).
static bool gen9_mul_mat(gen9_context * ctx, struct ggml_tensor * dst) {
    const struct ggml_tensor * w = dst->src[0], * x = dst->src[1];
    const int K = (int) w->ne[0], N = (int) w->ne[1], M = (int) x->ne[1];
    const int ldb = (int) (w->nb[1] / sizeof(float)), lda = (int) (x->nb[1] / sizeof(float)), ldc = (int) (dst->nb[1] / sizeof(float));
    std::lock_guard<std::mutex> lk(gpu_mu);
    if (is_weight(w)) {
        wkey k = { w->data, ggml_nbytes(w), (int) w->type, w->ne[0], w->ne[1] };
        auto it = ctx->cache.find(k);
        gen9blas_packed * P;
        if (it == ctx->cache.end()) {
            P = gen9blas_pack_b(GEN9BLAS_S, 1, K, N, w->data, ldb);
            if (!P) { fprintf(stderr, "gen9: pack_b %s failed: %s\n", w->name, gen9blas_error()); return false; }
            ctx->cache[k] = P; n_mm_packed++;
        } else { P = it->second; n_mm_cached++; }
        if (gen9blas_gemm_pb(GEN9BLAS_S, 0, M, 1.0, x->data, lda, P, 0.0, dst->data, ldc)) {
            fprintf(stderr, "gen9: gemm_pb %s failed: %s\n", w->name, gen9blas_error()); return false;
        }
        return true;
    }
    n_mm_percall++;
    if (gen9blas_gemm(GEN9BLAS_S, 0, 1, M, N, K, 1.0, x->data, lda, w->data, ldb, 0.0, dst->data, ldc)) {
        fprintf(stderr, "gen9: gemm failed: %s\n", gen9blas_error()); return false;
    }
    return true;
}

static enum ggml_status gen9_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    gen9_context * ctx = (gen9_context *) backend->context;
    n_graphs++;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];
        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) { n_noop++; continue; }
        switch (node->op) {
            case GGML_OP_NONE: case GGML_OP_RESHAPE: case GGML_OP_VIEW: case GGML_OP_PERMUTE: case GGML_OP_TRANSPOSE:
                n_noop++;
                break;
            case GGML_OP_MUL_MAT:
                if (!gen9_mul_mat(ctx, node)) return GGML_STATUS_FAILED;
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
static ggml_backend_t gen9_device_init_backend(ggml_backend_dev_t, const char *) {
    return new ggml_backend {
        /* .guid    = */ gen9_guid(),
        /* .iface   = */ gen9_backend_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_gen9_reg(), 0),
        /* .context = */ new gen9_context,
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

static bool fits_int(int64_t v) { return v > 0 && v <= INT_MAX; }

// Design section 7 (F32 weights only in this version; F16 needs new kernels).
static bool gen9_supports_mul_mat(const struct ggml_tensor * op) {
    const struct ggml_tensor * w = op->src[0], * x = op->src[1];
    n_mm_asked++;
    if (w->type != GGML_TYPE_F32 || x->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) { n_mm_declined_type++; return false; }
    if (w->ne[2] != 1 || w->ne[3] != 1 || x->ne[2] != 1 || x->ne[3] != 1 || !ggml_is_contiguous(w) || !ggml_is_contiguous(x) ||
        !ggml_is_contiguous(op) || !fits_int(w->ne[0]) || !fits_int(w->ne[1]) || !fits_int(x->ne[1])) { n_mm_declined_shape++; return false; }
    if (op->ne[0] < 32 || op->ne[1] < 32 || x->ne[0] < 32) { n_mm_declined_batch++; return false; }
    if (ggml_get_op_params_i32(op, 1) == GGML_HINT_SRC0_IS_HADAMARD) { n_mm_declined_hint++; return false; }
    if (!gpu_ok()) { n_mm_declined_gpu++; return false; }
    return true;
}

static bool gen9_dev_supports_op(ggml_backend_dev_t, const struct ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_NONE: case GGML_OP_RESHAPE: case GGML_OP_VIEW: case GGML_OP_PERMUTE: case GGML_OP_TRANSPOSE:
            return true;   // no-op nodes: test-backend-ops asks for every tensor of a test graph
        case GGML_OP_MUL_MAT:
            return gen9_supports_mul_mat(op);
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

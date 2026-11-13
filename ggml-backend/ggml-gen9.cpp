// ggml backend "GEN9" on gen9blas kernels (docs/ggml-backend-design.md). Step 1: MUL_MAT with F32 and F16 weights.
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
#include <cfloat>
#include <chrono>
#include <climits>
#include <thread>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <tuple>
#include <vector>
#include <unistd.h>

#ifndef GEN9_BENCH_DIR
#define GEN9_BENCH_DIR ""
#endif

// ---------------- routing counter (printed at exit with GEN9_INFO=1) ----------------

static std::atomic<long> n_graphs{0}, n_noop{0}, n_mm_cached{0}, n_mm_packed{0}, n_mm_percall{0}, n_mm_f16{0}, n_mm_f16_fallback{0},
                         n_ffn_fused{0}, n_ffn_fallback{0}, n_glu_alone{0}, n_glu_asked{0}, n_glu_declined{0};
static std::atomic<long> n_mm_asked{0}, n_mm_declined_shape{0}, n_mm_declined_type{0}, n_mm_declined_batch{0},
                         n_mm_declined_hint{0}, n_mm_declined_gpu{0};

static std::atomic<long> n_mm_streamed{0};
static std::atomic<long> n_dequant{0}, n_dequant_f32{0}, n_dequant_gpu{0};
static std::atomic<double> t_dequant_s{0}, t_deq_alloc_s{0}, t_deq_cpu_s{0}, t_deq_pack_s{0}, t_deq_gpu_s{0};
// FLASH_ATTN_EXT (opt-in GEN9_ATTN=1, docs/attention-gpu.md section 11)
static std::atomic<long> n_fa_asked{0}, n_fa_off{0}, n_fa_decl_type{0}, n_fa_decl_param{0}, n_fa_decl_shape{0}, n_fa_decl_layout{0},
                         n_fa_decl_nq{0}, n_fa_decl_mask{0}, n_fa_decl_gpu{0};
static std::atomic<long> n_fa_gpu{0}, n_fa_launch{0}, n_fa_rows{0}, n_fa_cpu_mask{0}, n_fa_cpu_rows{0}, n_fa_cpu_lib{0}, n_fa_cpu_err{0};
static std::atomic<double> t_fa_gpu_s{0}, t_fa_mask_s{0}, t_fa_cpu_s{0}, fa_flop{0};

static void gen9_report(void) {
    fprintf(stderr, "gen9: graph_compute %ld; no-op nodes %ld; MUL_MAT run: cached weight %ld, packed now %ld, per call %ld; "
            "F16 weights %ld (recomputed in F32 after FP16 overflow: %ld); weights over the cache budget, packed per use %ld\n",
            n_graphs.load(), n_noop.load(), n_mm_cached.load(), n_mm_packed.load(), n_mm_percall.load(), n_mm_f16.load(), n_mm_f16_fallback.load(), n_mm_streamed.load());
    fprintf(stderr, "gen9: FFN blocks fused %ld (recomputed unfused after FP16 overflow: %ld); SWIGLU alone %ld; "
            "supports_op GLU asked %ld, declined %ld\n", n_ffn_fused.load(), n_ffn_fallback.load(), n_glu_alone.load(),
            n_glu_asked.load(), n_glu_declined.load());
    fprintf(stderr, "gen9: quantized weights dequantized %ld (on the GPU: %ld; to F32 because of FP16 range: %ld), %.2f s "
            "(GPU %.2f; CPU path: alloc %.2f, to_float+FP16 %.2f, pack_b %.2f)\n", n_dequant.load(), n_dequant_gpu.load(),
            n_dequant_f32.load(), t_dequant_s.load(), t_deq_gpu_s.load(), t_deq_alloc_s.load(), t_deq_cpu_s.load(), t_deq_pack_s.load());
    fprintf(stderr, "gen9: supports_op MUL_MAT asked %ld, declined: shape/layout %ld, type %ld, batch<32 %ld, hint %ld, no GPU %ld\n",
            n_mm_asked.load(), n_mm_declined_shape.load(), n_mm_declined_type.load(), n_mm_declined_batch.load(),
            n_mm_declined_hint.load(), n_mm_declined_gpu.load());
    fprintf(stderr, "gen9: supports_op FLASH_ATTN_EXT asked %ld, declined: GEN9_ATTN off %ld, type %ld, ALiBi/softcap/sinks %ld, "
            "shape/no binary %ld, layout %ld, n_q<32 %ld, mask/kv_rows shape %ld, no GPU %ld\n", n_fa_asked.load(), n_fa_off.load(),
            n_fa_decl_type.load(), n_fa_decl_param.load(), n_fa_decl_shape.load(), n_fa_decl_layout.load(), n_fa_decl_nq.load(),
            n_fa_decl_mask.load(), n_fa_decl_gpu.load());
    fprintf(stderr, "gen9: FLASH_ATTN_EXT run on the GPU %ld (%ld launches, %ld with kv_rows), %.3f s, %.1f GFLOP (%.1f GFLOPS incl. "
            "pinning/submit/wait); mask analysis %.3f s; computed by ggml-cpu inside GEN9: mask not causal/full %ld, kv_rows not "
            "identity %ld, library declined %ld, GPU error %ld (%.3f s)\n", n_fa_gpu.load(), n_fa_launch.load(), n_fa_rows.load(),
            t_fa_gpu_s.load(), fa_flop.load() * 1e-9, t_fa_gpu_s.load() > 0 ? fa_flop.load() * 1e-9 / t_fa_gpu_s.load() : 0.0,
            t_fa_mask_s.load(), n_fa_cpu_mask.load(), n_fa_cpu_rows.load(), n_fa_cpu_lib.load(), n_fa_cpu_err.load(), t_fa_cpu_s.load());
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
struct packed_t { gen9blas_packed * P; int t; bool temp = false; };   // temp: not cached, free after use   // t: GEN9BLAS_S (F32 GEMM) or GEN9BLAS_HS (accurate FP16 GEMM)
struct gen9_context {
    std::map<wkey, packed_t> cache;
    size_t cached_bytes = 0;
    ggml_backend_t cpu = nullptr;   // ggml-cpu for FLASH_ATTN_EXT nodes whose mask turns out not to be supported (created on first use)
    ~gen9_context() {
        if (cpu) ggml_backend_free(cpu);
        std::lock_guard<std::mutex> lk(gpu_mu); for (auto & kv : cache) gen9blas_packed_free(kv.second.P);
    }
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

// Rows of w (any type with to_float: F16, quantized) as F32 or FP16 in a dense N x K buffer, 4 threads.
// Returns false if any value is not finite or (FP16 output) beyond the FP16 range.
static bool dequant_rows(const struct ggml_tensor * w, void * out, bool to_f16) {
    const int64_t K = w->ne[0], N = w->ne[1];
    const ggml_to_float_t to_float = ggml_get_type_traits(w->type)->to_float;
    std::atomic<bool> ok{true};
    auto work = [&](int64_t n0, int64_t n1) {
        std::vector<float> row(K);
        for (int64_t n = n0; n < n1; n++) {
            const void * src = (const char *) w->data + n * w->nb[1];
            float * f = to_f16 ? row.data() : (float *) out + n * K;
            to_float(src, f, K);
            for (int64_t i = 0; i < K; i++) if (!(fabsf(f[i]) <= (to_f16 ? 65504.0f : FLT_MAX))) { ok = false; break; }
            if (to_f16) ggml_fp32_to_fp16_row(f, (ggml_fp16_t *) out + n * K, K);
        }
    };
    const int nt = N >= 64 ? (int) std::min(8u, std::max(1u, std::thread::hardware_concurrency())) : 1;
    std::vector<std::thread> th;
    for (int i = 0; i < nt; i++) th.emplace_back(work, N * i / nt, N * (i + 1) / nt);
    for (auto & t : th) t.join();
    return ok;
}

// gen9blas GPU dequantization type for a ggml type, -1 if the GPU kernel does not handle it.
static int gpu_qtype(enum ggml_type t) {
    switch (t) {
        case GGML_TYPE_Q8_0: return GEN9BLAS_Q8_0;
        case GGML_TYPE_Q4_K: return GEN9BLAS_Q4_K;
        case GGML_TYPE_Q6_K: return GEN9BLAS_Q6_K;
        default: return -1;
    }
}

// Packs a weight for the GEMM: F32 -> SGEMM; F16 -> accurate FP16 GEMM; quantized -> dequantized to FP16 on the GPU (Q8_0, Q4_K,
// Q6_K; GEN9_CPU_DEQ=1: CPU) or on the CPU (step Q1, other types), or to F32 if any value does not fit FP16. Caller holds gpu_mu.
static packed_t pack_weight(const struct ggml_tensor * w) {
    const int K = (int) w->ne[0], N = (int) w->ne[1];
    if (w->type == GGML_TYPE_F32) return { gen9blas_pack_b(GEN9BLAS_S, 1, K, N, w->data, (int) (w->nb[1] / 4)), GEN9BLAS_S };
    if (w->type == GGML_TYPE_F16) return { gen9blas_pack_b(GEN9BLAS_HS, 1, K, N, w->data, (int) (w->nb[1] / 2)), GEN9BLAS_HS };
    auto now = [] { return std::chrono::steady_clock::now(); };
    auto secs = [](std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) { return std::chrono::duration<double>(b - a).count(); };
    auto t0 = now();
    n_dequant++;
    const int qt = gpu_qtype(w->type);
    if (qt >= 0 && !getenv("GEN9_CPU_DEQ")) {   // dequantized on the GPU straight into the packed layout (bit-identical)
        int oor = 0;
        gen9blas_packed * P = gen9blas_pack_b_q(qt, K, N, w->data, w->nb[1], &oor);
        if (P && !oor && getenv("GEN9_DEQ_VERIFY")) {   // debug: same GEMM with the CPU-path packing must give the same bits
            std::vector<ggml_fp16_t> h((size_t) N * K);
            dequant_rows(w, h.data(), true);
            gen9blas_packed * P2 = gen9blas_pack_b(GEN9BLAS_HS, 1, K, N, h.data(), K);
            const int M = 32;
            std::vector<float> A((size_t) M * K), C1((size_t) M * N), C2((size_t) M * N);
            for (size_t i = 0; i < A.size(); i++) A[i] = (float) ((i * 2654435761u) % 2001) / 1000.0f - 1.0f;
            int nf = 0;
            gen9blas_gemm_pb_f16w(M, A.data(), K, P, C1.data(), N, &nf);
            gen9blas_gemm_pb_f16w(M, A.data(), K, P2, C2.data(), N, &nf);
            size_t bad = 0, first = (size_t) -1;
            for (size_t i = 0; i < C1.size(); i++) if (memcmp(&C1[i], &C2[i], 4)) { bad++; if (first == (size_t) -1) first = i; }
            fprintf(stderr, "gen9: verify %-28s %s K=%d N=%d nb1=%zu data%%64=%zu: %zu of %zu differ (first col %zu)\n", w->name,
                    ggml_type_name(w->type), K, N, (size_t) w->nb[1], (size_t) ((uintptr_t) w->data % 64), bad, C1.size(),
                    first == (size_t) -1 ? 0 : first % N);
            gen9blas_packed_free(P2);
        }
        if (P && !oor) {
            n_dequant_gpu++;
            const double t = secs(t0, now());
            t_deq_gpu_s = t_deq_gpu_s + t; t_dequant_s = t_dequant_s + t;
            return { P, GEN9BLAS_HS };
        }
        if (P) gen9blas_packed_free(P);   // beyond FP16: the CPU path below packs it as F32
        else fprintf(stderr, "gen9: GPU dequantization of %s failed (%s); CPU path\n", w->name, gen9blas_error());
        t0 = now();
    }
    std::vector<ggml_fp16_t> h((size_t) N * K);
    packed_t r;
    auto t1 = now();
    const bool fits = dequant_rows(w, h.data(), true);
    auto t2 = now();
    t_deq_alloc_s = t_deq_alloc_s + secs(t0, t1); t_deq_cpu_s = t_deq_cpu_s + secs(t1, t2);
    if (fits) {
        r = { gen9blas_pack_b(GEN9BLAS_HS, 1, K, N, h.data(), K), GEN9BLAS_HS };
        t_deq_pack_s = t_deq_pack_s + secs(t2, now());
    } else {
        h.clear(); h.shrink_to_fit();
        std::vector<float> f((size_t) N * K);
        dequant_rows(w, f.data(), false);
        r = { gen9blas_pack_b(GEN9BLAS_S, 1, K, N, f.data(), K), GEN9BLAS_S };
        n_dequant_f32++;
    }
    t_dequant_s = t_dequant_s + secs(t0, now());
    return r;
}

// Budget for packed weights kept on the GPU (they live in system RAM next to llama's mmap of the model): GEN9_CACHE_MB,
// default 40 % of physical memory. Weights are cached in first-use order until the budget is full; the rest are packed
// again on every use and freed (a forward pass touches weights cyclically, so keeping the first ones beats LRU).
static size_t cache_budget(void) {
    static size_t v = 0;
    if (!v) {
        const char * e = getenv("GEN9_CACHE_MB");
        v = e && atof(e) > 0 ? (size_t) (atof(e) * 1048576.0) : (size_t) sysconf(_SC_PHYS_PAGES) * (size_t) sysconf(_SC_PAGE_SIZE) / 10 * 4;
    }
    return v;
}

// Packed weight from the cache (packed on first use), or packed for this use only (r.temp) when over the budget.
// Caller holds gpu_mu and frees a temp result with release().
static packed_t get_packed(gen9_context * ctx, const struct ggml_tensor * w) {
    wkey k = { w->data, ggml_nbytes(w), (int) w->type, w->ne[0], w->ne[1] };
    auto it = ctx->cache.find(k);
    if (it != ctx->cache.end()) { n_mm_cached++; return it->second; }
    const size_t bytes = (size_t) ((w->ne[1] + 15) / 16 * 16) * (size_t) ((w->ne[0] + 31) / 32 * 32) * (w->type == GGML_TYPE_F32 ? 4 : 2);
    packed_t r = pack_weight(w);
    if (!r.P) { fprintf(stderr, "gen9: pack_b %s failed: %s\n", w->name, gen9blas_error()); return r; }
    if (ctx->cached_bytes + bytes > cache_budget()) { r.temp = true; n_mm_streamed++; return r; }
    ctx->cache[k] = r; ctx->cached_bytes += bytes; n_mm_packed++;
    return r;
}

static void release(const packed_t & p) { if (p.temp && p.P) gen9blas_packed_free(p.P); }

// dst[M x N] = src1[M x K] . src0^T with src0 stored N x K row-major: gemm(A = src1, B = src0 transposed).
// F32 weights: SGEMM. F16 and quantized weights: accurate FP16 GEMM (gemm_pb_f16w); a call whose result overflows FP16 is
// recomputed with the weights converted to F32 (rare: largest chunk sum seen on Qwen2.5-1.5B is 25x below the limit).
static bool gen9_mul_mat(gen9_context * ctx, struct ggml_tensor * dst) {
    const struct ggml_tensor * w = dst->src[0], * x = dst->src[1];
    const int K = (int) w->ne[0], N = (int) w->ne[1], M = (int) x->ne[1];
    const int lda = (int) (x->nb[1] / sizeof(float)), ldc = (int) (dst->nb[1] / sizeof(float));
    std::lock_guard<std::mutex> lk(gpu_mu);
    packed_t p;
    bool temp = false;
    if (is_weight(w)) {
        p = get_packed(ctx, w);
        if (!p.P) return false;
        temp = p.temp;
    } else {
        n_mm_percall++;
        if (w->type == GGML_TYPE_F32) {
            if (gen9blas_gemm(GEN9BLAS_S, 0, 1, M, N, K, 1.0, x->data, lda, w->data, (int) (w->nb[1] / 4), 0.0, dst->data, ldc)) {
                fprintf(stderr, "gen9: gemm failed: %s\n", gen9blas_error()); return false;
            }
            return true;
        }
        p = pack_weight(w);
        if (!p.P) { fprintf(stderr, "gen9: pack_b failed: %s\n", gen9blas_error()); return false; }
        temp = true;
    }
    int rc;
    if (p.t == GEN9BLAS_S) rc = gen9blas_gemm_pb(GEN9BLAS_S, 0, M, 1.0, x->data, lda, p.P, 0.0, dst->data, ldc);
    else {
        int nonfinite = 0;
        n_mm_f16++;
        rc = gen9blas_gemm_pb_f16w(M, (const float *) x->data, lda, p.P, (float *) dst->data, ldc, &nonfinite);
        if (!rc && nonfinite) {   // FP16 overflow somewhere: recompute this call in F32
            n_mm_f16_fallback++;
            std::vector<float> wf((size_t) N * K);
            dequant_rows(w, wf.data(), false);
            rc = gen9blas_gemm(GEN9BLAS_S, 0, 1, M, N, K, 1.0, x->data, lda, wf.data(), K, 0.0, dst->data, ldc);
        }
    }
    if (temp) gen9blas_packed_free(p.P);
    if (rc) { fprintf(stderr, "gen9: MUL_MAT %s failed: %s\n", w->name, gen9blas_error()); return false; }
    return true;
}

// SWIGLU (split form: dst = silu(src0) * src1) on the CPU, for the rare node that is not fused (e.g. eval callbacks
// that hand over single nodes). Single-threaded: correct, not fast; the counter shows if it happens in normal runs.
static void gen9_glu_alone(struct ggml_tensor * dst) {
    const struct ggml_tensor * a = dst->src[0], * b = dst->src[1];
    const int64_t nc = dst->ne[0], nr = ggml_nrows(dst);
    for (int64_t r = 0; r < nr; r++) {
        const float * x = (const float *) ((const char *) a->data + r * a->nb[1]);
        const float * g = (const float *) ((const char *) b->data + r * b->nb[1]);
        float * y = (float *) ((char *) dst->data + r * dst->nb[1]);
        for (int64_t i = 0; i < nc; i++) y[i] = x[i] / (1.0f + expf(-x[i])) * g[i];
    }
    n_glu_alone++;
}

// FFN fusion is opt-in (GEN9_FFN=1): it is correct and saves GPU and CPU work, but on this 25 W machine the shared
// power budget moves to the GPU and the CPU attention slows down by as much (results/ggml-backend-ffn-fusion-2026-11-13.txt).
static bool ffn_off(void) { static const bool v = getenv("GEN9_FFN") == nullptr; return v; }

static bool is_f16_weight_mm(const struct ggml_tensor * n) {
    return n->op == GGML_OP_MUL_MAT && n->src[0]->type == GGML_TYPE_F16 && is_weight(n->src[0]);
}

// Gated FFN block at nodes i..i+3: two MUL_MATs on the same input (gate, up; either order), SWIGLU of the two, MUL_MAT
// (down) of the SWIGLU; F16 weights; the three intermediates used only inside the block (ggml_node_has_n_uses counts
// over the whole graph, also when this split is a view of it). Returns true if computed fused.
static bool gen9_try_ffn(gen9_context * ctx, struct ggml_cgraph * cg, int i) {
    if (ffn_off() || i + 3 >= cg->n_nodes) return false;
    struct ggml_tensor * m0 = cg->nodes[i], * m1 = cg->nodes[i + 1], * glu = cg->nodes[i + 2], * down = cg->nodes[i + 3];
    if (!is_f16_weight_mm(m0) || !is_f16_weight_mm(m1) || !is_f16_weight_mm(down)) return false;
    if (glu->op != GGML_OP_GLU || ggml_get_glu_op(glu) != GGML_GLU_OP_SWIGLU || !glu->src[1]) return false;
    if (m0->src[1] != m1->src[1] || down->src[1] != glu) return false;
    struct ggml_tensor * gate = glu->src[0], * up = glu->src[1];
    if (!((gate == m0 && up == m1) || (gate == m1 && up == m0))) return false;
    if (!ggml_node_has_n_uses(cg, i, 1) || !ggml_node_has_n_uses(cg, i + 1, 1) || !ggml_node_has_n_uses(cg, i + 2, 1)) return false;
    const struct ggml_tensor * x = m0->src[1];
    if (x->type != GGML_TYPE_F32 || !ggml_is_contiguous(x) || x->ne[2] != 1 || x->ne[3] != 1) return false;
    if (gate->src[0]->ne[1] != down->src[0]->ne[0] || up->src[0]->ne[1] != gate->src[0]->ne[1]) return false;
    std::lock_guard<std::mutex> lk(gpu_mu);
    const packed_t g = get_packed(ctx, gate->src[0]), u = get_packed(ctx, up->src[0]), d = get_packed(ctx, down->src[0]);
    struct rel3 { const packed_t & a, & b, & c; ~rel3() { release(a); release(b); release(c); } } rel{ g, u, d };
    if (!g.P || !u.P || !d.P || g.t != GEN9BLAS_HS || u.t != GEN9BLAS_HS || d.t != GEN9BLAS_HS) return false;
    gen9blas_packed * Pg = g.P, * Pu = u.P, * Pd = d.P;
    int nonfinite = 0;
    const int M = (int) x->ne[1];
    if (gen9blas_ffn_f16w(M, (const float *) x->data, (int) (x->nb[1] / sizeof(float)), Pg, Pu, Pd, (float *) down->data,
                          (int) (down->nb[1] / sizeof(float)), &nonfinite)) {
        fprintf(stderr, "gen9: ffn_f16w failed: %s\n", gen9blas_error()); return false;
    }
    if (nonfinite) { n_ffn_fallback++; return false; }   // the caller recomputes the four nodes one by one
    n_ffn_fused++;
    return true;
}

// ---------------- FLASH_ATTN_EXT (docs/attention-gpu.md section 11) ----------------
// Opt-in (GEN9_ATTN=1). supports_op accepts what the library's kernels take: F16 K/V, F32 Q/O in llama.cpp's layouts,
// no ALiBi / soft-capping / sinks, a (D, G, H_kv) with a built-in binary, n_q >= 32. The mask contents are only known at
// compute time: the kernel masks by index (token t sees the first vis0 + dvis*t keys), so the mask must be exactly that
// pattern (0 then -inf in every row; llama.cpp's single-sequence causal mask, or no mask); with kv_rows the listed rows
// must be the identity over the visible columns. Anything else (several sequences, holes after a context shift, SWA,
// a mask with other values) is computed by ggml-cpu inside this backend (counted).
static bool attn_on(void) { static const bool v = getenv("GEN9_ATTN") && atoi(getenv("GEN9_ATTN")) != 0; return v; }

static float fa_param(const struct ggml_tensor * op, int i) { float v; memcpy(&v, (const float *) op->op_params + i, sizeof v); return v; }

static bool gen9_supports_fa(const struct ggml_tensor * op) {
    n_fa_asked++;
    if (!attn_on()) { n_fa_off++; return false; }
    const struct ggml_tensor * q = op->src[0], * k = op->src[1], * v = op->src[2], * m = op->src[3], * rows = op->src[5];
    if (q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F16 || v->type != GGML_TYPE_F16 || op->type != GGML_TYPE_F32) { n_fa_decl_type++; return false; }
    if (op->src[4] || fa_param(op, 1) != 0.0f || fa_param(op, 2) != 0.0f) { n_fa_decl_param++; return false; }
    const int64_t D = q->ne[0], nq = q->ne[1], H = q->ne[2], nk = k->ne[1], HKV = k->ne[2];
    if (k->ne[0] != D || v->ne[0] != D || v->ne[1] != nk || v->ne[2] != HKV || q->ne[3] != 1 || k->ne[3] != 1 || v->ne[3] != 1 ||
        HKV <= 0 || H % HKV || nk % 16 || nk > INT_MAX || !gpu_ok() || !gen9blas_attn_supported((int) D, (int) H, (int) HKV)) {
        if (!gpu_ok()) n_fa_decl_gpu++; else n_fa_decl_shape++;
        return false;
    }
    const size_t row = (size_t) (HKV * D * 2);
    if (q->nb[0] != 4 || q->nb[1] != (size_t) (H * D * 4) || q->nb[2] != (size_t) (D * 4) || k->nb[0] != 2 || k->nb[1] != row ||
        k->nb[2] != (size_t) (D * 2) || v->nb[0] != 2 || v->nb[1] != row || v->nb[2] != (size_t) (D * 2) || !ggml_is_contiguous(op) ||
        op->ne[0] != D || op->ne[1] != H || op->ne[2] != nq || op->ne[3] != 1) { n_fa_decl_layout++; return false; }
    if (nq < 32) { n_fa_decl_nq++; return false; }
    const int64_t ncols = rows ? rows->ne[0] : nk;
    if ((m && (m->type != GGML_TYPE_F16 || !ggml_is_contiguous(m) || m->ne[0] != ncols || m->ne[1] < nq || m->ne[2] != 1 || m->ne[3] != 1 || ncols % 16)) ||
        (rows && (!m || rows->type != GGML_TYPE_I32 || !ggml_is_contiguous(rows) || rows->ne[1] != 1 || ncols > nk))) {
        n_fa_decl_mask++; return false;
    }
    return true;
}

// Mask rows 0..nq-1: row t must be ncols values, the first n_t equal to +0 and the rest -inf, with n_t = n_0 + dvis*t,
// dvis 0 or 1, n_0 >= 1. Word-wise compare (+0 = 0x0000, -inf = 0xFC00 in F16).
static bool fa_mask_prefix(const struct ggml_tensor * m, int64_t nq, int & vis0, int & dvis) {
    const int64_t nc = m->ne[0];
    auto ok_row = [&](const uint16_t * r, int64_t n) {
        int64_t j = 0;
        for (; j + 4 <= n; j += 4) { uint64_t w; memcpy(&w, r + j, 8); if (w) return false; }
        for (; j < n; j++) if (r[j]) return false;
        for (; j < nc && (j & 3); j++) if (r[j] != 0xFC00) return false;
        for (; j + 4 <= nc; j += 4) { uint64_t w; memcpy(&w, r + j, 8); if (w != 0xFC00FC00FC00FC00ull) return false; }
        for (; j < nc; j++) if (r[j] != 0xFC00) return false;
        return true;
    };
    auto row = [&](int64_t t) { return (const uint16_t *) ((const char *) m->data + t * m->nb[1]); };
    int64_t n0 = 0;
    while (n0 < nc && row(0)[n0] == 0) n0++;
    if (n0 < 1 || !ok_row(row(0), n0)) return false;
    int d = 0;
    if (nq > 1) {
        if (n0 < nc && ok_row(row(1), n0 + 1)) d = 1;
        else if (!ok_row(row(1), n0)) return false;
    }
    for (int64_t t = 2; t < nq; t++) if (!ok_row(row(t), std::min(n0 + d * t, nc))) return false;
    vis0 = (int) n0; dvis = d;
    return true;
}

static void fa_on_cpu(gen9_context * ctx, struct ggml_cgraph * cg, int i) {
    auto t0 = std::chrono::steady_clock::now();
    if (!ctx->cpu) {
        ctx->cpu = ggml_backend_cpu_init();
        const char * e = getenv("GEN9_CPU_THREADS");
        ggml_backend_cpu_set_n_threads(ctx->cpu, e && atoi(e) > 0 ? atoi(e) : (int) std::max(1u, std::thread::hardware_concurrency()));
    }
    struct ggml_cgraph one = *cg;   // = ggml_graph_view(cg, i, i + 1), which libggml-base does not export
    one.size = 0; one.n_nodes = 1; one.n_leafs = 0; one.nodes = cg->nodes + i;
    one.grads = nullptr; one.grad_accs = nullptr; one.leafs = nullptr; one.uid = 0;
    GGML_ASSERT(ggml_backend_graph_compute(ctx->cpu, &one) == GGML_STATUS_SUCCESS);
    t_fa_cpu_s = t_fa_cpu_s + std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

static void gen9_flash_attn(gen9_context * ctx, struct ggml_cgraph * cg, int i) {
    struct ggml_tensor * dst = cg->nodes[i];
    const struct ggml_tensor * q = dst->src[0], * k = dst->src[1], * v = dst->src[2], * m = dst->src[3], * rows = dst->src[5];
    const int D = (int) q->ne[0], nq = (int) q->ne[1], H = (int) q->ne[2], HKV = (int) k->ne[2];
    int vis0 = (int) k->ne[1], dvis = 0;
    auto t0 = std::chrono::steady_clock::now();
    if (m && !fa_mask_prefix(m, nq, vis0, dvis)) {
        t_fa_mask_s = t_fa_mask_s + std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        n_fa_cpu_mask++; fa_on_cpu(ctx, cg, i); return;
    }
    const int64_t vmax = std::min<int64_t>(vis0 + (int64_t) dvis * (nq - 1), rows ? rows->ne[0] : k->ne[1]);
    if (rows) {   // kv_rows: column c reads cache row rows[c]; identity over the visible columns = a plain view
        const int32_t * r = (const int32_t *) rows->data;
        for (int64_t c = 0; c < vmax; c++) if (r[c] != c) { n_fa_cpu_rows++; fa_on_cpu(ctx, cg, i); return; }
        n_fa_rows++;
    }
    auto t1 = std::chrono::steady_clock::now();
    t_fa_mask_s = t_fa_mask_s + std::chrono::duration<double>(t1 - t0).count();
    const int nk = (int) std::min<int64_t>((vmax + 15) / 16 * 16, k->ne[1]);   // key rows the kernel reads (multiple of 16)
    // pin the whole KV tensor behind the view (its first row is the view's first row in llama.cpp)
    auto reg_bytes = [](const struct ggml_tensor * t) {
        const struct ggml_tensor * b = t->view_src ? t->view_src : t;
        return (size_t) ((const char *) b->data + ggml_nbytes(b) - (const char *) t->data);
    };
    int launches = 0, rc;
    {
        std::lock_guard<std::mutex> lk(gpu_mu);
        rc = gen9blas_attn_f16kv(D, H, HKV, nq, nk, (const float *) q->data, (const uint16_t *) k->data, reg_bytes(k),
                                 (const uint16_t *) v->data, reg_bytes(v), (float *) dst->data, fa_param(dst, 0), vis0, dvis, &launches);
    }
    if (rc) {
        if (rc < 0) { n_fa_cpu_err++; fprintf(stderr, "gen9: attention on the GPU failed (%s); ggml-cpu\n", gen9blas_attn_error()); }
        else n_fa_cpu_lib++;
        fa_on_cpu(ctx, cg, i); return;
    }
    double fl = 0;
    for (int t = 0; t < nq; t++) fl += 4.0 * H * D * std::min<int64_t>(vis0 + (int64_t) dvis * t, nk);
    n_fa_gpu++; n_fa_launch += launches; fa_flop = fa_flop + fl;
    t_fa_gpu_s = t_fa_gpu_s + std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
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
                if (gen9_try_ffn(ctx, cgraph, i)) { i += 3; break; }
                if (!gen9_mul_mat(ctx, node)) return GGML_STATUS_FAILED;
                break;
            case GGML_OP_GLU:
                gen9_glu_alone(node);
                break;
            case GGML_OP_FLASH_ATTN_EXT:
                gen9_flash_attn(ctx, cgraph, i);
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

// Own buffer type "GEN9_Host": ordinary CPU memory in the standard ggml layout (is_host), allocated by the CPU buffer
// type. It only exists so that llama.cpp puts it first in the weight list of an ACCEL device (llama-model.cpp:1234-1243
// skips the plain CPU type): weights whose MUL_MAT GEN9 accepts then stay in the standard layout instead of going to
// ggml-cpu's REPACK buffer, which GEN9 cannot read (Q4_K etc. were all left on the CPU before). With mmap, llama.cpp still
// wraps the file through buffer_from_host_ptr (a plain CPU buffer). The CPU reads these weights as any host buffer.
static const char * gen9_buft_get_name(ggml_backend_buffer_type_t) { return "GEN9_Host"; }
static ggml_backend_buffer_type_t gen9_buft(void);
static ggml_backend_buffer_t gen9_buft_alloc(ggml_backend_buffer_type_t, size_t size) {
    ggml_backend_buffer_t b = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
    if (b) b->buft = gen9_buft();
    return b;
}
static size_t gen9_buft_get_alignment(ggml_backend_buffer_type_t) { return ggml_backend_buft_get_alignment(ggml_backend_cpu_buffer_type()); }
static bool gen9_buft_is_host(ggml_backend_buffer_type_t) { return true; }

ggml_backend_reg_t ggml_backend_gen9_reg(void);
static ggml_backend_buffer_type_t gen9_buft(void) {
    static struct ggml_backend_buffer_type t = {
        /* .iface = */ {
            /* .get_name         = */ gen9_buft_get_name,
            /* .alloc_buffer     = */ gen9_buft_alloc,
            /* .alloc_buffer_n   = */ NULL,
            /* .get_alignment    = */ gen9_buft_get_alignment,
            /* .get_max_size     = */ NULL,
            /* .get_alloc_size   = */ NULL,
            /* .get_alloc_size_n = */ NULL,
            /* .is_host          = */ gen9_buft_is_host,
        },
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_gen9_reg(), 0),
        /* .context = */ NULL,
    };
    return &t;
}

static ggml_backend_buffer_type_t gen9_dev_get_buffer_type(ggml_backend_dev_t) {
    return getenv("GEN9_CPU_BUFT") ? ggml_backend_cpu_buffer_type() : gen9_buft();   // GEN9_CPU_BUFT: old behaviour (A/B)
}

static ggml_backend_buffer_t gen9_dev_buffer_from_host_ptr(ggml_backend_dev_t, void * ptr, size_t size, size_t) {
    return ggml_backend_cpu_buffer_from_ptr(ptr, size);
}

static bool fits_int(int64_t v) { return v > 0 && v <= INT_MAX; }

// Design section 7: F32 or F16 weights, F32 activations and result.
static bool gen9_supports_mul_mat(const struct ggml_tensor * op) {
    const struct ggml_tensor * w = op->src[0], * x = op->src[1];
    n_mm_asked++;
    const bool w_ok = w->type == GGML_TYPE_F32 || w->type == GGML_TYPE_F16 ||
                      (ggml_is_quantized(w->type) && ggml_get_type_traits(w->type)->to_float != nullptr);   // Q1: dequantized
    if (!w_ok || x->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
        n_mm_declined_type++; return false;
    }
    if (w->ne[2] != 1 || w->ne[3] != 1 || x->ne[2] != 1 || x->ne[3] != 1 || !ggml_is_contiguous(w) || !ggml_is_contiguous(x) ||
        !ggml_is_contiguous(op) || !fits_int(w->ne[0]) || !fits_int(w->ne[1]) || !fits_int(x->ne[1])) { n_mm_declined_shape++; return false; }
    if (op->ne[0] < 32 || op->ne[1] < 32 || x->ne[0] < 32) { n_mm_declined_batch++; return false; }
    if (ggml_get_op_params_i32(op, 1) == GGML_HINT_SRC0_IS_HADAMARD) { n_mm_declined_hint++; return false; }
    if (w->type != GGML_TYPE_F32) {   // FP16 chunk sums and FP16 activations: not allowed if F32 / BF16 precision is required
        const enum ggml_prec acc = ggml_prec_get_acc(op), p1 = ggml_prec_get_src(op, 1);
        if (acc == GGML_PREC_F32 || acc == GGML_PREC_BF16 || p1 == GGML_PREC_F32 || p1 == GGML_PREC_BF16) { n_mm_declined_hint++; return false; }
    }
    if (!gpu_ok()) { n_mm_declined_gpu++; return false; }
    return true;
}

static bool gen9_dev_supports_op(ggml_backend_dev_t, const struct ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_NONE: case GGML_OP_RESHAPE: case GGML_OP_VIEW: case GGML_OP_PERMUTE: case GGML_OP_TRANSPOSE:
            return true;   // no-op nodes: test-backend-ops asks for every tensor of a test graph
        case GGML_OP_MUL_MAT:
            return gen9_supports_mul_mat(op);
        case GGML_OP_FLASH_ATTN_EXT:
            return gen9_supports_fa(op);
        case GGML_OP_GLU: {   // SWIGLU of two MUL_MAT results: computed inside a fused FFN block (gen9_try_ffn)
            n_glu_asked++;
            const struct ggml_tensor * a = op->src[0], * b = op->src[1];
            const bool ok = ggml_get_glu_op(op) == GGML_GLU_OP_SWIGLU && b && a->type == GGML_TYPE_F32 && b->type == GGML_TYPE_F32 &&
                            op->type == GGML_TYPE_F32 && ggml_is_contiguous(a) && ggml_is_contiguous(b) && ggml_is_contiguous(op) &&
                            ggml_are_same_shape(a, b) && a->ne[2] == 1 && a->ne[3] == 1 && op->ne[1] >= 32 &&
                            a->op == GGML_OP_MUL_MAT && b->op == GGML_OP_MUL_MAT && a->src[0]->type == GGML_TYPE_F16 &&
                            b->src[0]->type == GGML_TYPE_F16 && !ffn_off() && gpu_ok();   // only what gen9_try_ffn can fuse
            if (!ok) n_glu_declined++;
            return ok;
        }
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

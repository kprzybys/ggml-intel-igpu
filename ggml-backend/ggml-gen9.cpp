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
#include "gen9blas_ml.h"

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
#include <set>
#include <string>
#include <unordered_map>
#include <tuple>
#include <vector>
#include <unistd.h>


// ---------------- routing counter (printed at exit with GEN9_INFO=1) ----------------

static std::atomic<long> n_graphs{0}, n_noop{0}, n_mm_cached{0}, n_mm_packed{0}, n_mm_percall{0}, n_mm_f16{0}, n_mm_f16_fallback{0},
                         n_ffn_fused{0}, n_ffn_fallback{0}, n_glu_alone{0}, n_glu_asked{0}, n_glu_declined{0};
static std::atomic<long> n_mm_asked{0}, n_mm_declined_shape{0}, n_mm_declined_type{0}, n_mm_declined_batch{0},
                         n_mm_declined_hint{0}, n_mm_declined_gpu{0};

static std::atomic<long> n_mm_streamed{0};
static std::atomic<long> n_multi_groups{0}, n_multi_mm{0}, n_multi_overlap{0}, n_multi_fallback{0};
static std::atomic<long> n_ew_cpu_splits{0}, n_ew_cpu_nodes{0};
static std::atomic<long> n_ew_norm{0}, n_ew_norm_wb{0}, n_ew_gelu{0}, n_mm_f16in{0};
static std::atomic<long> n_ew_rms{0}, n_ew_rms_mul{0}, n_ew_add{0}, n_ew_mul{0}, n_ew_rope{0}, n_ew_sync_graphs{0}, n_ew_asked{0}, n_ew_declined{0};
static std::atomic<long> n_dequant{0}, n_dequant_f32{0}, n_dequant_gpu{0};
static std::atomic<long> n_epi{0}, n_epi_gelu{0}, n_epi_res{0}, n_epi_res_alias{0}, n_epi_fallback{0};
static std::atomic<long> n_normpack{0}, n_pa_used{0};
static std::atomic<double> t_dequant_s{0}, t_deq_alloc_s{0}, t_deq_cpu_s{0}, t_deq_pack_s{0}, t_deq_gpu_s{0};
// FLASH_ATTN_EXT (opt-in GEN9_ATTN=1, docs/attention-gpu.md section 11)
static std::atomic<long> n_fa_asked{0}, n_fa_off{0}, n_fa_decl_type{0}, n_fa_decl_param{0}, n_fa_decl_shape{0}, n_fa_decl_layout{0},
                         n_fa_decl_nq{0}, n_fa_decl_mask{0}, n_fa_decl_gpu{0};
static std::atomic<long> n_fa_gpu{0}, n_fa_launch{0}, n_fa_rows{0}, n_fa_cpu_mask{0}, n_fa_cpu_rows{0}, n_fa_cpu_lib{0}, n_fa_cpu_err{0};
static std::atomic<long> n_fa_streams{0}, n_fa_streams_cpu{0}, n_fa_verify{0}, n_fa_dec{0};
static double fa_verify_max = 0;
static std::atomic<long> n_fa_f16gather{0}, n_ew_setrows{0}, n_nw_splits{0}, n_nw_overflow{0};
// GEN9_EW_VISION=0: CPY/CONT/DUP, UPSCALE and ROPE mode VISION stay on ggml-cpu (mtmd vision encoders, PaddleOCR-VL)
static bool vision_ops_on(void) { static const bool v = [] { const char * e = getenv("GEN9_EW_VISION"); return !(e && !strcmp(e, "0")); }(); return v; }
static bool mrope_on(void) { static const bool v = [] { const char * e = getenv("GEN9_EW_MROPE"); return e && !strcmp(e, "1"); }(); return v; }   // opt-in
static bool mm_nowait_on(void) { static const bool v = [] { const char * e = getenv("GEN9_MM_NOWAIT"); return e && !strcmp(e, "1"); }(); return v; }   // FLASH_ATTN_EXT with F16 K/V gathered from another layout (fa_f16_gather_off)
static std::atomic<long> n_fa_f32kv{0}, n_mm_forced_f32{0}, n_ew_im2col{0}, n_ew_cpy{0}, n_ew_cpy_cpu{0}, n_ew_upscale{0}, n_ew_vrope{0};   // FLASH_ATTN_EXT with F32 K/V converted to FP16 (NeMo)
static std::atomic<long> n_fa_tail{0}, n_fa_cpu_tail{0};   // n_k not a multiple of 16: rounded up on the GPU / V tail not finite
static std::atomic<long> n_fa_blk{0}, n_fa_blk_runs{0}, n_fa_blk_gather{0}, n_fa_cpu_blk{0};   // general masks (fa_mask_runs)
static std::atomic<double> t_fa_gpu_s{0}, t_fa_mask_s{0}, t_fa_cpu_s{0}, fa_flop{0};
// GEN9_PROF=1: wall time per graph_compute branch, with a wait after every elementwise op (as GEN9_EW_SYNC). GEMM and
// attention shares are right; elementwise ops carry their own wait (~0.2 ms each), which a normal run hides, so their
// share is overstated and the total is slower: compare PROF runs only with PROF runs (kcl's KCL_PHASES=1 for submit/wait).
enum { PR_FFN, PR_QKV, PR_EPI, PR_MM, PR_GLU, PR_RMS, PR_BINOP, PR_ROPE, PR_NORM, PR_GELU, PR_FA, PR_EWCPU, PR_N };
static const char * pr_name[PR_N] = { "FFN fused", "Q/K/V group", "MUL_MAT+epilogue", "MUL_MAT", "GLU alone", "RMS_NORM(+MUL)",
                                      "ADD/MUL", "ROPE", "NORM", "GELU", "FLASH_ATTN", "elementwise on ggml-cpu" };
static std::atomic<double> pr_t[PR_N], pr_flop[PR_N], pr_graph_s{0};
static std::atomic<long> pr_n[PR_N];
static bool prof_on(void) { static const bool v = getenv("GEN9_PROF") && atoi(getenv("GEN9_PROF")) != 0; return v; }

static void gen9_report(void) {
    fprintf(stderr, "gen9: FLASH_ATTN_EXT with KV streams on the GPU %ld (decoding kernel %ld), back to ggml-cpu %ld; GEN9_ATTN_VERIFY nodes %ld, max NMSE %.3e\n", n_fa_streams.load() + n_fa_dec.load(), n_fa_dec.load(), n_fa_streams_cpu.load(), n_fa_verify.load(), fa_verify_max);
    fprintf(stderr, "gen9: no-wait splits (GEN9_MM_NOWAIT) %ld, failed on FP16 overflow %ld\n", n_nw_splits.load(), n_nw_overflow.load());
    fprintf(stderr, "gen9: MUL_MAT groups on one input (Q/K/V) %ld covering %ld MUL_MATs; candidates refused for memory overlap %ld; "
            "outputs recomputed in F32 after FP16 overflow %ld\n", n_multi_groups.load(), n_multi_mm.load(), n_multi_overlap.load(),
            n_multi_fallback.load());
    fprintf(stderr, "gen9: splits with elementwise ops but no MUL_MAT / attention (their ops run on ggml-cpu inside GEN9) %ld, nodes %ld\n",
            n_ew_cpu_splits.load(), n_ew_cpu_nodes.load());
    fprintf(stderr, "gen9: whisper-type ops on the GPU: NORM %ld (with MUL+ADD fused %ld), GELU %ld; MUL_MAT with F16 activations %ld\n",
            n_ew_norm.load(), n_ew_norm_wb.load(), n_ew_gelu.load(), n_mm_f16in.load());
    fprintf(stderr, "gen9: MUL_MAT + bias fused in the epilogue %ld (+ GELU %ld, + residual %ld; residual left unfused because it "
            "aliases the output %ld; FP16 overflow -> unfused %ld)\n", n_epi.load(), n_epi_gelu.load(), n_epi_res.load(),
            n_epi_res_alias.load(), n_epi_fallback.load());
    fprintf(stderr, "gen9: LayerNorm writing the packed GEMM input %ld; MUL_MATs using it (no own packing) %ld\n",
            n_normpack.load(), n_pa_used.load());
    fprintf(stderr, "gen9: elementwise ops on the GPU: RMS_NORM %ld (with MUL fused %ld), ADD %ld, MUL %ld, ROPE %ld, SET_ROWS %ld; splits run with a "
            "wait per op (span not registered or GEN9_EW_SYNC) %ld; supports_op asked %ld, declined %ld\n", n_ew_rms.load(),
            n_ew_rms_mul.load(), n_ew_add.load(), n_ew_mul.load(), n_ew_rope.load(), n_ew_setrows.load(), n_ew_sync_graphs.load(), n_ew_asked.load(),
            n_ew_declined.load());
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
    if (n_mm_forced_f32) fprintf(stderr, "gen9: MUL_MAT forced to F32 (GEN9_MM_F32=1) %ld\n", n_mm_forced_f32.load());
    fprintf(stderr, "gen9: IM2COL on the GPU %ld; CPY/CONT/DUP on the GPU %ld (misaligned -> ggml-cpu %ld); UPSCALE %ld; ROPE vision %ld\n",
            n_ew_im2col.load(), n_ew_cpy.load(), n_ew_cpy_cpu.load(), n_ew_upscale.load(), n_ew_vrope.load());
    fprintf(stderr, "gen9: FLASH_ATTN_EXT with F32 K/V (converted to FP16 in the staging buffer) on the GPU %ld; F16 K/V gathered from another layout %ld\n", n_fa_f32kv.load(), n_fa_f16gather.load());
    fprintf(stderr, "gen9: FLASH_ATTN_EXT run on the GPU %ld (%ld launches, %ld with kv_rows), %.3f s, %.1f GFLOP (%.1f GFLOPS incl. "
            "pinning/submit/wait); mask analysis %.3f s; computed by ggml-cpu inside GEN9: mask not causal/full %ld, kv_rows not "
            "identity %ld, library declined %ld, GPU error %ld (%.3f s)\n", n_fa_gpu.load(), n_fa_launch.load(), n_fa_rows.load(),
            t_fa_gpu_s.load(), fa_flop.load() * 1e-9, t_fa_gpu_s.load() > 0 ? fa_flop.load() * 1e-9 / t_fa_gpu_s.load() : 0.0,
            t_fa_mask_s.load(), n_fa_cpu_mask.load(), n_fa_cpu_rows.load(), n_fa_cpu_lib.load(), n_fa_cpu_err.load(), t_fa_cpu_s.load());
    fprintf(stderr, "gen9: FLASH_ATTN_EXT with n_k not a multiple of 16: rounded up on the GPU %ld, V tail not finite -> ggml-cpu %ld\n",
            n_fa_tail.load(), n_fa_cpu_tail.load());
    fprintf(stderr, "gen9: FLASH_ATTN_EXT general masks (several sequences, wrapped cache, kv_rows not identity): on the GPU %ld "
            "(%ld runs, %ld with K/V gathered), computed by ggml-cpu %ld\n", n_fa_blk.load(), n_fa_blk_runs.load(),
            n_fa_blk_gather.load(), n_fa_cpu_blk.load());
    if (prof_on()) {
        double sum = 0; for (int k = 0; k < PR_N; k++) sum += pr_t[k].load();
        fprintf(stderr, "gen9: PROF graph_compute %.3f s, of it in the branches below %.3f s (rest: loop, no-op nodes)\n",
                pr_graph_s.load(), sum);
        for (int k = 0; k < PR_N; k++) {
            if (!pr_n[k].load()) continue;
            const double t = pr_t[k].load(), f = pr_flop[k].load();
            fprintf(stderr, "gen9: PROF %-24s %7ld calls %9.3f s %5.1f %%", pr_name[k], pr_n[k].load(), t,
                    pr_graph_s.load() > 0 ? 100 * t / pr_graph_s.load() : 0.0);
            if (f > 0) fprintf(stderr, "  %9.1f GFLOP %6.1f GFLOPS", f * 1e-9, t > 0 ? f * 1e-9 / t : 0.0);
            fprintf(stderr, "\n");
        }
    }
}

// ---------------- GPU (lazy: first use) ----------------

static std::mutex gpu_mu;   // gen9blas keeps global state (work buffers, one queue)

static bool gpu_ok(void) {
    static int state = -1;   // -1 not tried, 0 failed, 1 ready
    std::lock_guard<std::mutex> lk(gpu_mu);
    if (state < 0) {
        state = gen9blas_init() == 0;
        if (!state) fprintf(stderr, "gen9: GPU init failed (%s); declining everything\n", gen9blas_error());
    }
    return state == 1;
}

// ---------------- backend (stream) ----------------

struct wkey {
    const void * data; size_t nbytes; int type; int64_t ne0, ne1;
    bool operator<(const wkey & o) const { return std::tie(data, nbytes, type, ne0, ne1) < std::tie(o.data, o.nbytes, o.type, o.ne0, o.ne1); }
};

// The cache lives until the last backend instance is freed (llama.cpp frees it with the context, before the model
// (design section 3.1): a data pointer cannot be reused for another weight while an entry exists.
struct packed_t { gen9blas_packed_b * P; int t; bool temp = false; };   // temp: not cached, free after use   // t: GEN9BLAS_S (F32 GEMM) or GEN9BLAS_HS (accurate FP16 GEMM)
// Packed weights are shared by every backend instance of the process (whisper.cpp -p N: one instance per state, same
// model): one cache, freed by the last instance. Guarded by gpu_mu like the rest of the library state.
struct gen9_wcache { std::map<wkey, packed_t> map; size_t bytes = 0; int refs = 0; };
static gen9_wcache g_wcache;
struct gen9_context {
    std::map<wkey, packed_t> & cache = g_wcache.map;
    size_t & cached_bytes = g_wcache.bytes;
    gen9_context() { std::lock_guard<std::mutex> lk(gpu_mu); g_wcache.refs++; }
    bool ew_sync = false, ew_pending = false, ew_cpu = false;   // ew_cpu: split without a waiting GPU call: elementwise ops on ggml-cpu
    bool nw = false;   // this split runs its products in gen9blas no-wait mode (GEN9_MM_NOWAIT=1, all MUL_MATs with M <= 16)   // ew_cpu: split without a waiting GPU call: elementwise ops on ggml-cpu
    std::vector<char> done;   // per graph_compute: nodes already computed by a MUL_MAT group   // per graph_compute: wait after each elementwise op / ops queued, not waited for
    ggml_backend_t cpu = nullptr;   // ggml-cpu for FLASH_ATTN_EXT nodes whose mask turns out not to be supported (created on first use)
    // packed GEMM inputs written by LayerNorm (gen9blas_layer_norm_pack), per graph_compute; two buffers in turn: reusing
    // one drops its map entries, so a late consumer packs for itself (correct in any node order)
    struct pa_entry { gen9blas_packed_a * pa; const void * data; };
    std::map<const struct ggml_tensor *, pa_entry> pa_map;
    gen9blas_packed_a * pa_pool[2] = { nullptr, nullptr };
    int pa_next = 0;
    ~gen9_context() {
        for (auto * pa : pa_pool) gen9blas_packed_a_free(pa);
        if (cpu) ggml_backend_free(cpu);
        std::lock_guard<std::mutex> lk(gpu_mu);
        if (--g_wcache.refs == 0) { for (auto & kv : cache) gen9blas_packed_b_free(kv.second.P); cache.clear(); cached_bytes = 0; }
    }
};

// Address inside a read-only, file-backed mapping (/proc/self/maps "r-.." with a path): such memory cannot change, so a
// packed copy of a tensor there stays valid. NeMo-Speech.cpp maps its GGUF this way without marking the buffer WEIGHTS.
// Ranges re-read when an address is not found (a model mapped later); GEN9_RO_WEIGHTS=0 turns this detection off.
static bool in_ro_file_map(const void * p) {
    static std::mutex mu; static std::vector<std::pair<uintptr_t, uintptr_t>> ro; static std::unordered_map<const void *, bool> memo;
    static const bool off = [] { const char * e = getenv("GEN9_RO_WEIGHTS"); return e && !strcmp(e, "0"); }();
    if (off || !p) return false;
    std::lock_guard<std::mutex> lk(mu);
    auto m = memo.find(p);
    if (m != memo.end()) return m->second;
    const uintptr_t a = (uintptr_t) p;
    auto find = [&] { for (auto & r : ro) if (a >= r.first && a < r.second) return true; return false; };
    bool hit = find();
    if (!hit) {
        ro.clear();
        if (FILE * f = fopen("/proc/self/maps", "r")) {
            char line[1024];
            while (fgets(line, sizeof line, f)) {
                unsigned long lo, hi; char perm[8]; unsigned long off_; char dev[16]; unsigned long ino; char path[768] = "";
                if (sscanf(line, "%lx-%lx %7s %lx %15s %lu %767s", &lo, &hi, perm, &off_, dev, &ino, path) >= 6 &&
                    perm[0] == 'r' && perm[1] == '-' && path[0] == '/' && ino != 0)
                    ro.push_back({ lo, hi });
            }
            fclose(f);
        }
        hit = find();
    }
    memo[p] = hit;
    return hit;
}
static bool is_weight(const struct ggml_tensor * w) {
    if (w->op != GGML_OP_NONE || w->view_src != nullptr || (w->flags & (GGML_TENSOR_FLAG_INPUT | GGML_TENSOR_FLAG_PARAM)) != 0 || !w->buffer)
        return false;
    return w->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS || in_ro_file_map(w->data);
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
        gen9blas_packed_b * P = gen9blas_pack_b_q(qt, K, N, w->data, w->nb[1], &oor);
        if (P && !oor && getenv("GEN9_DEQ_VERIFY")) {   // debug: same GEMM with the CPU-path packing must give the same bits
            std::vector<ggml_fp16_t> h((size_t) N * K);
            dequant_rows(w, h.data(), true);
            gen9blas_packed_b * P2 = gen9blas_pack_b(GEN9BLAS_HS, 1, K, N, h.data(), K);
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
            gen9blas_packed_b_free(P2);
        }
        if (P && !oor) {
            n_dequant_gpu++;
            const double t = secs(t0, now());
            t_deq_gpu_s = t_deq_gpu_s + t; t_dequant_s = t_dequant_s + t;
            return { P, GEN9BLAS_HS };
        }
        if (P) gen9blas_packed_b_free(P);   // beyond FP16: the CPU path below packs it as F32
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

static void release(const packed_t & p) { if (p.temp && p.P) { gen9blas_flush(); gen9blas_packed_b_free(p.P); } }   // flush: no-wait work may still read it

// dst[M x N] = src1[M x K] . src0^T with src0 stored N x K row-major: gemm(A = src1, B = src0 transposed).
// F32 weights: SGEMM. F16 and quantized weights: accurate FP16 GEMM (gemm_pb_f16w); a call whose result overflows FP16 is
// recomputed with the weights converted to F32 (rare: largest chunk sum seen on Qwen2.5-1.5B is 25x below the limit).
static bool gen9_mul_mat(gen9_context * ctx, struct ggml_tensor * dst) {
    const struct ggml_tensor * w = dst->src[0], * x = dst->src[1];
    const int K = (int) w->ne[0], N = (int) w->ne[1], M = (int) x->ne[1];
    const bool xh = x->type == GGML_TYPE_F16;   // FP16 activations (ggml conv_1d: F16 kernel as src1, im2col as src0)
    const int lda = (int) (x->nb[1] / ggml_type_size(x->type)), ldc = (int) (dst->nb[1] / sizeof(float));
    std::vector<float> xf;   // F32 copy of FP16 activations, only for the F32 paths
    auto x_f32 = [&]() -> const float * {
        if (!xh) return (const float *) x->data;
        if (xf.empty()) { if (ctx->ew_pending) { gen9blas_flush(); ctx->ew_pending = false; }   // x read on the CPU; gpu_mu held
            xf.resize((size_t) M * K); for (int r = 0; r < M; r++) ggml_fp16_to_fp32_row((const ggml_fp16_t *) x->data + (size_t) r * lda, xf.data() + (size_t) r * K, K); }
        return xf.data();
    };
    const int ldf = xh ? K : lda;
    if (xh) n_mm_f16in++;
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
            if (gen9blas_gemm(GEN9BLAS_S, 0, 1, M, N, K, 1.0, x_f32(), ldf, w->data, (int) (w->nb[1] / 4), 0.0, dst->data, ldc)) {
                fprintf(stderr, "gen9: gemm failed: %s\n", gen9blas_error()); return false;
            }
            return true;
        }
        p = pack_weight(w);
        if (!p.P) { fprintf(stderr, "gen9: pack_b failed: %s\n", gen9blas_error()); return false; }
        temp = true;
    }
    int rc;
    if (p.t == GEN9BLAS_S) rc = gen9blas_gemm_pb(0, M, 1.0, x_f32(), ldf, p.P, 0.0, dst->data, ldc);
    else {
        int nonfinite = 0;
        n_mm_f16++;
        auto pit = xh ? ctx->pa_map.end() : ctx->pa_map.find(x);
        if (pit != ctx->pa_map.end() && pit->second.data == x->data) {   // input packed by the LayerNorm that produced it
            n_pa_used++;
            rc = gen9blas_gemm_pa_f16w_ex(pit->second.pa, p.P, (float *) dst->data, ldc, nullptr, 0, nullptr, 0, &nonfinite);
        } else
            rc = xh ? gen9blas_gemm_pb_f16w_h(M, (const uint16_t *) x->data, lda, p.P, (float *) dst->data, ldc, &nonfinite)
                    : gen9blas_gemm_pb_f16w(M, (const float *) x->data, lda, p.P, (float *) dst->data, ldc, &nonfinite);
        if (!rc && nonfinite) {   // FP16 overflow somewhere: recompute this call in F32
            n_mm_f16_fallback++;
            std::vector<float> wf((size_t) N * K);
            dequant_rows(w, wf.data(), false);
            rc = gen9blas_gemm(GEN9BLAS_S, 0, 1, M, N, K, 1.0, x_f32(), ldf, wf.data(), K, 0.0, dst->data, ldc);
        }
    }
    if (temp) { if (ctx->nw) gen9blas_flush(); gen9blas_packed_b_free(p.P); }   // no-wait: the GPU may still read it
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

// FFN fusion is on by default since 2026-11-13 evening (GEN9_FFN=0 turns it off). It was opt-in while attention ran on
// the CPU and lost the shared 25 W budget to the GPU (results/ggml-backend-ffn-fusion-2026-11-13.txt); with GPU attention
// and quantized weights accepted it is faster in every pair and not costlier in energy
// (results/ggml-backend-ffn-quantized-2026-11-13.txt).
static bool ffn_off(void) { static const bool v = [] { const char * e = getenv("GEN9_FFN"); return e && !strcmp(e, "0"); }(); return v; }

// Weights the fused FFN takes: F16, or a quantized type with to_float (dequantized into the same HS packed layout; a
// weight that falls back to F32 because of the FP16 range is rejected later by the g/u/d .t == HS check). Until
// 2026-11-13 only F16 was accepted, so GEN9_FFN=1 never fused on Q4_K_M / Q8_0 models (GEN9_INFO "fused 0").
static bool ffn_weight_type(enum ggml_type t) {
    return t == GGML_TYPE_F16 || (ggml_is_quantized(t) && ggml_get_type_traits(t)->to_float != nullptr);
}

static bool is_f16_weight_mm(const struct ggml_tensor * n) {
    return n->op == GGML_OP_MUL_MAT && ffn_weight_type(n->src[0]->type) && is_weight(n->src[0]);
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
    gen9blas_packed_b * Pg = g.P, * Pu = u.P, * Pd = d.P;
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

static void gen9_shape(const struct ggml_tensor * t, long ne[4], long nb[4]) {
    for (int k = 0; k < 4; k++) { ne[k] = (long) t->ne[k]; nb[k] = (long) t->nb[k]; }
}

// ---------------- FLASH_ATTN_EXT (docs/attention-gpu.md section 11) ----------------
// Opt-in (GEN9_ATTN=1). supports_op accepts what the library's kernels take: F16 K/V, F32 Q/O in llama.cpp's layouts,
// no ALiBi / soft-capping / sinks, a (D, G, H_kv) with a built-in binary, n_q >= 32. The mask contents are only known at
// compute time: the kernel masks by index (token t sees the first vis0 + dvis*t keys), so the mask must be exactly that
// pattern (0 then -inf in every row; llama.cpp's single-sequence causal mask, or no mask); with kv_rows the listed rows
// must be the identity over the visible columns. Anything else (several sequences, holes after a context shift, SWA,
// a mask with other values) is computed by ggml-cpu inside this backend (counted).
// n_k need not be a multiple of 16 when there is no mask and the KV tensor behind the views has the rows up to the next
// multiple (whisper.cpp's encoder: 1500 keys, buffer of 1536 rows); see fa_rows_behind.
// On by default since 2026-11-13 (measured on HD 530 and UHD 620, docs/attention-gpu.md section 11); GEN9_ATTN=0 turns it off.
static bool attn_on(void) { static const bool v = !getenv("GEN9_ATTN") || atoi(getenv("GEN9_ATTN")) != 0; return v; }

static float fa_param(const struct ggml_tensor * op, int i) { float v; memcpy(&v, (const float *) op->op_params + i, sizeof v); return v; }

// Key rows of K / V readable behind the view: rows up to the end of the tensor the view points into (view_src is the
// root tensor in ggml). A key count that is not a multiple of 16 (whisper.cpp's encoder: n_k = 1500 in a KV buffer
// padded to 1536 rows and zeroed at creation) is rounded up to 16 when these rows exist; the kernel masks keys >= vis0
// by substitution (-1e30 into the score lanes, asm/attn_gen.py mask()), so whatever K holds there is harmless, but
// p = 0 times a non-finite V value is NaN: the V tail is checked at compute time (gen9_flash_attn).
static int64_t fa_rows_behind(const struct ggml_tensor * t, size_t row) {
    if (!t->view_src) return t->ne[1];
    return (int64_t) ((ggml_nbytes(t->view_src) - t->view_offs) / row);
}

// F32 K/V (NeMo-Speech.cpp's RoPE encoder: K = RoPE output, V = a view of the fused QKV projection, both permuted):
// no mask, no kv_rows; any K/V row and head strides with contiguous D. gen9_flash_attn_f32kv converts them to FP16 in
// the kernel's layout ([key][head][d], rows padded with zeros to a multiple of 16) in the staging buffer. Q and the
// output in llama.cpp's layouts, as for F16 K/V. On by default; GEN9_ATTN_F32KV=0 turns it off. With the conversion on
// the CPU (one thread, under gpu_mu) it was slower than ggml-cpu (gp: 1 h 111 -> 134 s); on the GPU (gen9blas_cvt_f16_heads)
// and with n_q not divisible by 4 handled (NeMo's chunks of 265 / 530 rows were declined before), 1 h 99 -> 81 s.
// F16 K/V in another layout than llama.cpp's KV cache (mtmd's vision encoders: K/V cast to F16 from permuted views,
// [head][key][d]; PaddleOCR-VL): same staging path, the rows gathered by the CPU (memcpy of D halves per key and head;
// ~20 MB per layer for PaddleOCR-VL, a few ms against ~0.5 s of attention). GEN9_ATTN_F16GATHER=0 turns it off.
static bool fa_f16_gather_off(void) { static const bool v = [] { const char * e = getenv("GEN9_ATTN_F16GATHER"); return e && !strcmp(e, "0"); }(); return v; }

static bool fa_f32kv_ok(const struct ggml_tensor * op) {
    static const bool off32 = [] { const char * e = getenv("GEN9_ATTN_F32KV"); return e && !strcmp(e, "0"); }();
    const struct ggml_tensor * q = op->src[0], * k = op->src[1], * v = op->src[2];
    const bool off = k->type == GGML_TYPE_F16 ? fa_f16_gather_off() : off32;
    if (off || q->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32 || op->src[3] || op->src[4] || op->src[5] ||
        fa_param(op, 1) != 0.0f || fa_param(op, 2) != 0.0f) { n_fa_decl_type++; return false; }
    const int64_t D = q->ne[0], nq = q->ne[1], H = q->ne[2], nk = k->ne[1], HKV = k->ne[2];
    if (k->ne[0] != D || v->ne[0] != D || v->ne[1] != nk || v->ne[2] != HKV || q->ne[3] != 1 || k->ne[3] != 1 || v->ne[3] != 1 ||
        HKV <= 0 || H % HKV || nk > INT_MAX || !gpu_ok() || !gen9blas_attn_supported((int) D, (int) H, (int) HKV)) {
        if (!gpu_ok()) n_fa_decl_gpu++; else n_fa_decl_shape++;
        return false;
    }
    const size_t es = k->type == GGML_TYPE_F16 ? 2 : 4;
    if (q->nb[0] != 4 || q->nb[1] != (size_t) (H * D * 4) || q->nb[2] != (size_t) (D * 4) || k->nb[0] != es || v->nb[0] != es ||
        !ggml_is_contiguous(op) || op->ne[0] != D || op->ne[1] != H || op->ne[2] != nq || op->ne[3] != 1) { n_fa_decl_layout++; return false; }
    if (nq < 32) { n_fa_decl_nq++; return false; }
    return true;
}

// GEN9_INFO: each declined FLASH_ATTN_EXT shape once (D, H, H_kv, n_q, n_k, mask, kv_rows, K/V type and strides)
static void fa_shape_info(const struct ggml_tensor * op, const char * why) {
    if (!getenv("GEN9_INFO")) return;
    static std::set<std::string> seen;
    const struct ggml_tensor * q = op->src[0], * k = op->src[1], * v = op->src[2];
    char b[400];
    snprintf(b, sizeof b, "gen9: FA declined (%s): D %ld H %ld HKV %ld n_q %ld n_k %ld q.ne3 %ld k.ne3 %ld mask %s rows %s K %s nb %zu/%zu/%zu V nb %zu/%zu/%zu",
             why, (long) q->ne[0], (long) q->ne[2], (long) k->ne[2], (long) q->ne[1], (long) k->ne[1], (long) q->ne[3], (long) k->ne[3],
             op->src[3] ? "yes" : "no", op->src[5] ? "yes" : "no", ggml_type_name(k->type), k->nb[1], k->nb[2], k->nb[3], v->nb[1], v->nb[2], v->nb[3]);
    if (seen.insert(b).second) fprintf(stderr, "%s\n", b);
}

// KV streams (llama-server -np N, ggml ne[3] = streams): F16 K/V in llama.cpp's per-stream cache layout, one F16 mask per
// stream (or one for all), no kv_rows; any n_q (decoding: 1 per stream). gen9blas_attn_f16kv_streams. Opt-in: GEN9_ATTN_STREAMS=1
// (no measured gain on whole PaddleOCR-VL pages yet: kp-docsearch docs/plan-paddle-gen9.md).
// Decoding (n_q <= 32 per stream, GEN9_ATTN_DEC_MAXQ; D 128, G <= 8; measured: n_q 32 2.4x ggml-cpu, 64 / 128 slower): gen9blas_attn_dec reads K/V in the cache layout in place (no transposed
// copy), split over keys; on by default, GEN9_ATTN_DEC=0 turns it off. Larger n_q: the streams kernel above (opt-in).
static bool fa_dec_on(void) { static const bool v = [] { const char * e = getenv("GEN9_ATTN_DEC"); return !(e && !strcmp(e, "0")); }(); return v; }
static const int64_t FA_DEC_MAXQ = [] { const char * e = getenv("GEN9_ATTN_DEC_MAXQ"); return (int64_t) (e && atoi(e) > 0 ? atoi(e) : 32); }();
static bool fa_dec_ok(const struct ggml_tensor * op) {
    const struct ggml_tensor * q = op->src[0], * k = op->src[1], * v = op->src[2], * m = op->src[3];
    const int64_t D = q->ne[0], nq = q->ne[1], H = q->ne[2], ns = q->ne[3], nk = k->ne[1], HKV = k->ne[2];
    return fa_dec_on() && nq <= FA_DEC_MAXQ && D == 128 && HKV > 0 && H % HKV == 0 && gpu_ok() && gen9blas_attn_dec_supported((int) D, (int) H, (int) HKV) &&
        k->ne[3] == ns && v->ne[3] == ns && k->ne[0] == D && v->ne[0] == D && v->ne[1] == nk && v->ne[2] == HKV && nk <= INT_MAX && ns <= 64 &&
        !op->src[5] && q->nb[0] == 4 && q->nb[1] % 4 == 0 && q->nb[2] % 4 == 0 && q->nb[3] % 4 == 0 &&
        k->nb[0] == 2 && k->nb[1] % 16 == 0 && k->nb[2] % 16 == 0 && k->nb[3] % 2 == 0 &&
        v->nb[0] == 2 && v->nb[1] % 16 == 0 && v->nb[2] % 16 == 0 && v->nb[3] % 2 == 0 &&
        ggml_is_contiguous(op) && op->ne[0] == D && op->ne[1] == H && op->ne[2] == nq && op->ne[3] == ns &&
        m && m->type == GGML_TYPE_F16 && m->nb[0] == 2 && m->ne[0] == nk && m->ne[1] >= nq && m->ne[2] == 1 &&
        (m->ne[3] == ns || m->ne[3] == 1);
}
static bool fa_streams_ok(const struct ggml_tensor * op) {
    static const bool off = [] { const char * e = getenv("GEN9_ATTN_STREAMS"); return !(e && !strcmp(e, "1")); }();   // opt-in
    if (fa_dec_ok(op)) return true;
    const struct ggml_tensor * q = op->src[0], * k = op->src[1], * v = op->src[2], * m = op->src[3];
    const int64_t D = q->ne[0], nq = q->ne[1], H = q->ne[2], ns = q->ne[3], nk = k->ne[1], HKV = k->ne[2];
    const size_t row = (size_t) (HKV * D * 2);
    const bool ok = !off && k->ne[3] == ns && v->ne[3] == ns && k->ne[0] == D && v->ne[0] == D && v->ne[1] == nk && v->ne[2] == HKV &&
        HKV > 0 && H % HKV == 0 && nk <= INT_MAX && nk % 16 == 0 && !op->src[5] && gpu_ok() &&
        gen9blas_attn_streams_supported((int) D, (int) H, (int) HKV) &&
        q->nb[0] == 4 && q->nb[1] == (size_t) (H * D * 4) && q->nb[2] == (size_t) (D * 4) && q->nb[3] % 16 == 0 &&
        k->nb[0] == 2 && k->nb[1] == row && k->nb[2] == (size_t) (D * 2) && k->nb[3] % 16 == 0 &&
        v->nb[0] == 2 && v->nb[1] == row && v->nb[2] == (size_t) (D * 2) && v->nb[3] % 16 == 0 &&
        ggml_is_contiguous(op) && op->ne[0] == D && op->ne[1] == H && op->ne[2] == nq && op->ne[3] == ns && op->nb[3] == q->nb[3] &&
        m && m->type == GGML_TYPE_F16 && m->nb[0] == 2 && m->ne[0] == nk && m->ne[1] >= nq && m->ne[2] == 1 &&
        (m->ne[3] == ns || m->ne[3] == 1);
    if (!ok) { n_fa_decl_shape++; fa_shape_info(op, "streams"); }
    return ok;
}

static bool gen9_supports_fa(const struct ggml_tensor * op) {
    n_fa_asked++;
    if (!attn_on()) { n_fa_off++; return false; }
    const struct ggml_tensor * q = op->src[0], * k = op->src[1], * v = op->src[2], * m = op->src[3], * rows = op->src[5];
    if (k->type == GGML_TYPE_F32 && v->type == GGML_TYPE_F32) return fa_f32kv_ok(op);
    if (q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F16 || v->type != GGML_TYPE_F16 || op->type != GGML_TYPE_F32) { n_fa_decl_type++; return false; }
    if (op->src[4] || fa_param(op, 1) != 0.0f || fa_param(op, 2) != 0.0f) { n_fa_decl_param++; return false; }
    const int64_t D = q->ne[0], nq = q->ne[1], H = q->ne[2], nk = k->ne[1], HKV = k->ne[2];
    if (q->ne[3] > 1 || k->ne[3] > 1) return fa_streams_ok(op);
    if (k->ne[0] != D || v->ne[0] != D || v->ne[1] != nk || v->ne[2] != HKV || q->ne[3] != 1 || k->ne[3] != 1 || v->ne[3] != 1 ||
        HKV <= 0 || H % HKV || nk > INT_MAX || !gpu_ok() || !gen9blas_attn_supported((int) D, (int) H, (int) HKV)) {
        if (!gpu_ok()) n_fa_decl_gpu++; else n_fa_decl_shape++;
        fa_shape_info(op, "shape/no binary");
        return false;
    }
    const size_t row = (size_t) (HKV * D * 2);
    if (!m && !rows && k->nb[0] == 2 && v->nb[0] == 2 &&
        (k->nb[1] != row || k->nb[2] != (size_t) (D * 2) || v->nb[1] != row || v->nb[2] != (size_t) (D * 2)))
        return fa_f32kv_ok(op);   // F16 K/V in another layout: gathered into the staging buffer
    if (nk % 16) {   // only without a mask (a mask has n_k columns) and with the rounded-up rows behind both views
        const int64_t nk16 = (nk + 15) / 16 * 16;
        if (m || rows || fa_rows_behind(k, row) < nk16 || fa_rows_behind(v, row) < nk16) { n_fa_decl_shape++; fa_shape_info(op, "n_k % 16 with mask/rows"); return false; }
    }
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
static bool fa_mask_prefix_at(const void * mdata, size_t nb1, int64_t nc, int64_t nq, int & vis0, int & dvis);
static bool fa_mask_prefix(const struct ggml_tensor * m, int64_t nq, int & vis0, int & dvis) {
    return fa_mask_prefix_at(m->data, m->nb[1], m->ne[0], nq, vis0, dvis);
}
static bool fa_mask_prefix_at(const void * mdata, size_t nb1, int64_t nc, int64_t nq, int & vis0, int & dvis) {
    auto ok_row = [&](const uint16_t * r, int64_t n) {
        int64_t j = 0;
        for (; j + 4 <= n; j += 4) { uint64_t w; memcpy(&w, r + j, 8); if (w) return false; }
        for (; j < n; j++) if (r[j]) return false;
        for (; j < nc && (j & 3); j++) if (r[j] != 0xFC00) return false;
        for (; j + 4 <= nc; j += 4) { uint64_t w; memcpy(&w, r + j, 8); if (w != 0xFC00FC00FC00FC00ull) return false; }
        for (; j < nc; j++) if (r[j] != 0xFC00) return false;
        return true;
    };
    auto row = [&](int64_t t) { return (const uint16_t *) ((const char *) mdata + t * nb1); };
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

// General mask (several sequences in one ubatch, a ring KV cache that wrapped, kv_rows that are not the identity):
// rows t0..t0+nq-1 form a run when every mask value is +0 or -inf and each row sees the previous row's columns plus
// exactly one more (dvis 1) or the same ones (dvis 0). Softmax attention does not depend on the order of the keys, so
// a run is the single-sequence pattern over the key order "columns of the run's first row, then each newly visible
// column": col[] lists them; vis0 = |first row|. One launch per run (gen9_flash_attn_runs).
struct fa_run { int t0, nq, vis0, dvis; std::vector<int32_t> col; };
static bool fa_mask_runs(const struct ggml_tensor * m, int64_t nq, std::vector<fa_run> & runs) {
    const int64_t nc = m->ne[0];
    runs.clear();
    auto row = [&](int64_t t) { return (const uint16_t *) ((const char *) m->data + t * m->nb[1]); };
    for (int64_t t = 0; t < nq; t++) {
        const uint16_t * r = row(t), * p = t > 0 ? row(t - 1) : nullptr;
        fa_run * cur = runs.empty() ? nullptr : &runs.back();
        if (cur) {   // fast path: compare with the previous row 8 bytes at a time; the previous row was validated already, so
                     // only changed columns need a value check. Exactly one change -inf -> 0 extends a dvis 1 run, none a dvis 0 run.
            int64_t added = -1, ndiff = 0, c = 0;
            for (; c + 4 <= nc && ndiff < 2; c += 4) {
                if (c + 16 <= nc) {   // 32 bytes per step while equal
                    uint64_t x = 0;
                    for (int w = 0; w < 4; w++) { uint64_t a, b; memcpy(&a, r + c + 4 * w, 8); memcpy(&b, p + c + 4 * w, 8); x |= a ^ b; }
                    if (!x) { c += 12; continue; }
                }
                uint64_t a, b; memcpy(&a, r + c, 8); memcpy(&b, p + c, 8);
                if (a == b) continue;
                for (int64_t e = c; e < c + 4; e++) if (r[e] != p[e]) { ndiff++; added = (r[e] == 0 && p[e] == 0xFC00) ? e : -2; }
            }
            for (; c < nc && ndiff < 2; c++) if (r[c] != p[c]) { ndiff++; added = (r[c] == 0 && p[c] == 0xFC00) ? c : -2; }
            if (ndiff == 0 && (cur->nq == 1 || cur->dvis == 0)) { cur->dvis = 0; cur->nq++; continue; }
            if (ndiff == 1 && added >= 0 && (cur->nq == 1 || cur->dvis == 1)) {
                cur->dvis = 1; cur->col.push_back((int32_t) added); cur->nq++; continue;
            }
        }
        int64_t nz = 0;   // new run: full scan with value check
        for (int64_t c = 0; c < nc; c++) {
            const uint16_t x = r[c];
            if (x != 0 && x != 0xFC00) return false;
            if (x == 0) nz++;
        }
        if (nz < 1) return false;
        fa_run n{ (int) t, 1, (int) nz, 1, {} };
        n.col.reserve(nz + 64);
        for (int64_t c = 0; c < nc; c++) if (r[c] == 0) n.col.push_back((int32_t) c);
        runs.push_back(std::move(n));
    }
    return true;
}

static void ew_flush(gen9_context * ctx);
static void cpu_node(gen9_context * ctx, struct ggml_cgraph * cg, int i) {   // one node on ggml-cpu (inputs ready)
    ew_flush(ctx);   // inputs may come from queued elementwise ops
    if (!ctx->cpu) {
        ctx->cpu = ggml_backend_cpu_init();
        const char * e = getenv("GEN9_CPU_THREADS");
        ggml_backend_cpu_set_n_threads(ctx->cpu, e && atoi(e) > 0 ? atoi(e) : (int) std::max(1u, std::thread::hardware_concurrency()));
    }
    struct ggml_cgraph one = *cg;   // = ggml_graph_view(cg, i, i + 1), which libggml-base does not export
    one.size = 0; one.n_nodes = 1; one.n_leafs = 0; one.nodes = cg->nodes + i;
    one.grads = nullptr; one.grad_accs = nullptr; one.leafs = nullptr; one.uid = 0;
    GGML_ASSERT(ggml_backend_graph_compute(ctx->cpu, &one) == GGML_STATUS_SUCCESS);
}
// Mask (src[3]) and kv_rows (src[5]) are read on the CPU: wait for queued GPU work when either is computed by a node
// (in llama.cpp 2cbe0ec both are leaf inputs; a mask made with ggml_cast would be a queued CPY since 2026-11-15).
static void fa_cpu_inputs(gen9_context * ctx, const struct ggml_tensor * dst) {
    for (int s : { 3, 5 }) {
        const struct ggml_tensor * t = dst->src[s];
        if (t && (t->view_src ? t->view_src : t)->op != GGML_OP_NONE) { ew_flush(ctx); return; }
    }
}
static void fa_on_cpu(gen9_context * ctx, struct ggml_cgraph * cg, int i) {
    auto t0 = std::chrono::steady_clock::now();
    cpu_node(ctx, cg, i);
    t_fa_cpu_s = t_fa_cpu_s + std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// One launch per run of fa_mask_runs. A run whose key order maps to consecutive cache rows reads K/V in place (Q/O
// offset by t0 rows, K/V by the first row); otherwise its K/V rows are gathered into a reused, page-aligned staging
// buffer padded with zero rows to a multiple of 16. Any run the library cannot
// take -> the whole node on ggml-cpu (it overwrites what earlier runs wrote). kcl reuses a pinned object only for the
// exact same page range, so the varying ranges here (K/V from cr[0] to the tensor's end, staging nk rows) are pinned
// again; KCL_USERPTR_MB=2048 measured +1 % (results/attn-qwen3-prof-gp-2026-11-14.txt): left as is for now.
static void gen9_flash_attn_runs(gen9_context * ctx, struct ggml_cgraph * cg, int i, const std::vector<fa_run> & runs) {
    ew_flush(ctx);   // K/V rows are read on the CPU below: queued GPU writes (SET_ROWS into the cache) must land first
    static char * stage = nullptr; static size_t stage_bytes = 0;   // [K rows | V rows], used under gpu_mu
    struct ggml_tensor * dst = cg->nodes[i];
    const struct ggml_tensor * q = dst->src[0], * k = dst->src[1], * v = dst->src[2], * rows = dst->src[5];
    const int D = (int) q->ne[0], H = (int) q->ne[2], HKV = (int) k->ne[2];
    const size_t row = (size_t) HKV * D * 2, qrow = (size_t) H * D * 4;
    const int64_t behind = std::min(fa_rows_behind(k, row), fa_rows_behind(v, row));
    const int32_t * rr = rows ? (const int32_t *) rows->data : nullptr;
    const struct ggml_tensor * kroot = k->view_src ? k->view_src : k, * vroot = v->view_src ? v->view_src : v;
    auto t1 = std::chrono::steady_clock::now();
    int launches = 0; double fl = 0; long gathered = 0;
    for (const fa_run & r : runs) {
        const int64_t nv = (int64_t) r.col.size();             // keys the run's last row sees
        const int64_t nk = (nv + 15) / 16 * 16;
        std::vector<int64_t> cr(nv);
        bool direct = true;
        for (int64_t j = 0; j < nv; j++) {
            cr[j] = rr ? rr[r.col[j]] : r.col[j];
            if (cr[j] < 0 || cr[j] >= behind) { n_fa_cpu_blk++; fa_on_cpu(ctx, cg, i); return; }
            if (cr[j] != cr[0] + j) direct = false;
        }
        if (direct && cr[0] + nk > behind) direct = false;
        if (direct)   // rows behind the visible ones (to the 16-row rounding): p = 0, but 0 * Inf/NaN in V is NaN
            for (int64_t j = cr[0] + nv; j < cr[0] + nk && direct; j++) {
                const uint16_t * vr = (const uint16_t *) ((const char *) v->data + j * v->nb[1]);
                for (int64_t x = 0; x < (int64_t) HKV * D; x++) if ((vr[x] & 0x7C00) == 0x7C00) { direct = false; break; }
            }
        int nl = 0, rc = 1;
        {
            std::lock_guard<std::mutex> lk(gpu_mu);
            const char * kb = nullptr, * vb = nullptr; size_t kreg = 0, vreg = 0;
            if (direct) {
                kb = (const char *) k->data + (size_t) cr[0] * row; vb = (const char *) v->data + (size_t) cr[0] * row;
                kreg = (size_t) ((const char *) kroot->data + ggml_nbytes(kroot) - kb);
                vreg = (size_t) ((const char *) vroot->data + ggml_nbytes(vroot) - vb);
            } else {
                const size_t need = 2 * (size_t) nk * row;
                if (need > stage_bytes) {   // grow in MB steps; the old pinning stays in kcl's LRU until evicted
                    free(stage); stage = nullptr; stage_bytes = 0;
                    const size_t sz = (need + (1u << 20) - 1) & ~(size_t) ((1u << 20) - 1);
                    if (posix_memalign((void **) &stage, 4096, sz) == 0) stage_bytes = sz; else stage = nullptr;
                }
                if (stage) {
                    char * ks = stage, * vs = stage + (size_t) nk * row;
                    for (int64_t j = 0; j < nv; j++) {
                        memcpy(ks + j * row, (const char *) k->data + cr[j] * k->nb[1], row);
                        memcpy(vs + j * row, (const char *) v->data + cr[j] * v->nb[1], row);
                    }
                    memset(ks + nv * row, 0, (size_t) (nk - nv) * row); memset(vs + nv * row, 0, (size_t) (nk - nv) * row);
                    kb = ks; vb = vs; kreg = vreg = (size_t) nk * row; gathered++;
                }
            }
            if (kb)
                rc = gen9blas_attn_f16kv(D, H, HKV, r.nq, (int) nk, (const float *) ((const char *) q->data + (size_t) r.t0 * qrow),
                                         (const uint16_t *) kb, kreg, (const uint16_t *) vb, vreg,
                                         (float *) ((char *) dst->data + (size_t) r.t0 * qrow), fa_param(dst, 0), r.vis0, r.dvis, &nl);
        }
        if (rc) {
            if (rc < 0) { n_fa_cpu_err++; fprintf(stderr, "gen9: attention on the GPU failed (%s); ggml-cpu\n", gen9blas_error()); }
            n_fa_cpu_blk++; fa_on_cpu(ctx, cg, i); return;
        }
        launches += nl;
        for (int t = 0; t < r.nq; t++) fl += 4.0 * H * D * std::min<int64_t>(r.vis0 + (int64_t) r.dvis * t, nk);
    }
    n_fa_gpu++; n_fa_blk++; n_fa_blk_runs += (long) runs.size(); n_fa_blk_gather += gathered; n_fa_launch += launches; fa_flop = fa_flop + fl;
    if (rows) n_fa_rows++;
    t_fa_gpu_s = t_fa_gpu_s + std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
}

// F16 K/V in another layout: gathered on the GPU (gen9blas_cpy_any, queued behind the ops that wrote K/V, no wait) unless
// GEN9_ATTN_GPU_GATHER=0 (then on the CPU with a finiteness check, as before 2026-11-15). The GPU gather does not check
// for Inf/NaN. Not exact: an Inf in K can give a score of -Inf (p = 0) and a finite ggml-cpu row, while the kernel needs
// finite K/V; needs |K| > 65504 after the F32 -> F16 cast, not seen in the encoders served (Paddle's K is normalised).
// The padding rows nv..nk (only when n_k % 16 != 0) are cleared on the CPU first, so masked keys always meet finite V.
static bool fa_gpu_gather(void) { static const bool v = [] { const char * e = getenv("GEN9_ATTN_GPU_GATHER"); return !(e && !strcmp(e, "0")); }(); return v; }
static void gen9_flash_attn_f32kv(gen9_context * ctx, struct ggml_cgraph * cg, int i) {
    const bool f16 = cg->nodes[i]->src[1]->type == GGML_TYPE_F16, gg = f16 && fa_gpu_gather() && gen9blas_has(GEN9BLAS_FEAT_CPY);
    if (f16 && !gg) ew_flush(ctx);   // F16 K/V gathered on the CPU (queued SET_ROWS / CPY first)
    static char * stage = nullptr; static size_t stage_bytes = 0;   // [K rows | V rows | pad], used under gpu_mu
    struct ggml_tensor * dst = cg->nodes[i];
    const struct ggml_tensor * q = dst->src[0], * k = dst->src[1], * v = dst->src[2];
    const int D = (int) q->ne[0], nq = (int) q->ne[1], H = (int) q->ne[2], HKV = (int) k->ne[2], nv = (int) k->ne[1];
    const int nk = (nv + 15) / 16 * 16;
    const size_t row = (size_t) HKV * D * 2;
    auto t1 = std::chrono::steady_clock::now();
    int launches = 0, rc = 1;
    {
        std::lock_guard<std::mutex> lk(gpu_mu);
        const size_t need = 2 * (size_t) nk * row + 8192;   // + the kernel's read-ahead past the end of K^T (attn_gen.py: >= NK*D*16 B)
        if (need > stage_bytes) {
            if (ctx->ew_pending) {   // a queued gather of an earlier call may still use the old buffer
                if (gen9blas_flush()) fprintf(stderr, "gen9: flush failed: %s\n", gen9blas_error());
                ctx->ew_pending = false;
            }
            free(stage); stage = nullptr; stage_bytes = 0;
            const size_t sz = (need + (1u << 20) - 1) & ~(size_t) ((1u << 20) - 1);
            if (posix_memalign((void **) &stage, 4096, sz) == 0) { stage_bytes = sz; memset(stage, 0, sz); } else stage = nullptr;
        }
        if (stage) {
            ggml_fp16_t * ks = (ggml_fp16_t *) stage, * vs = (ggml_fp16_t *) (stage + (size_t) nk * row);
            // conversion on the GPU (gen9blas_cvt_f16_heads, saturated: no Inf), queued before the attention, which waits.
            // The staging buffer is a kcl span so the conversion's writes and the attention's reads use one GPU object
            // (else: wait after the conversion). Rows nv..nk keep older contents: always finite (zeroed at allocation or
            // saturated FP16 from an earlier call), so masked keys get p = 0 and 0 * V stays 0.
            bool fin = true;
            const bool span = gen9blas_span(stage, stage_bytes, 1) == 0;
            if (gg) {   // gather [key][head][d] rows on the GPU
                if (nk > nv) { memset(ks + (size_t) nv * HKV * D, 0, (size_t) (nk - nv) * row); memset(vs + (size_t) nv * HKV * D, 0, (size_t) (nk - nv) * row); }
                const long dnb[4] = { 2, (long) row, (long) D * 2, (long) nk * (long) row };
                long kne[4], knb[4], vne[4], vnb[4];
                gen9_shape(k, kne, knb); gen9_shape(v, vne, vnb);
                if (gen9blas_cpy_any(k->data, GEN9BLAS_H, kne, knb, ks, GEN9BLAS_H, kne, dnb, !span) || gen9blas_cpy_any(v->data, GEN9BLAS_H, vne, vnb, vs, GEN9BLAS_H, vne, dnb, !span)) {
                    fprintf(stderr, "gen9: K/V gather on the GPU failed: %s\n", gen9blas_error()); fin = false;
                }
                if (span) ctx->ew_pending = true;   // queued: a declined attention or the split end must still wait
            } else if (k->type == GGML_TYPE_F16) {   // gather [key][head][d] rows on the CPU (see fa_f16_gather_off)
                for (int t = 0; t < nv; t++)
                    for (int h = 0; h < HKV; h++) {
                        memcpy(ks + ((size_t) t * HKV + h) * D, (const char *) k->data + t * k->nb[1] + h * k->nb[2], (size_t) D * 2);
                        memcpy(vs + ((size_t) t * HKV + h) * D, (const char *) v->data + t * v->nb[1] + h * v->nb[2], (size_t) D * 2);
                    }
                for (int64_t x = 0; x < (int64_t) nv * HKV * D && fin; x++)   // the kernel needs finite K/V (no Inf/NaN)
                    if ((ks[x] & 0x7C00) == 0x7C00 || (vs[x] & 0x7C00) == 0x7C00) fin = false;
                if (!fin) {   // keep the staging invariant (rows nv..nk of a later call must be finite): clear what was copied
                    rc = 1;
                    memset(ks, 0, (size_t) nv * HKV * D * 2); memset(vs, 0, (size_t) nv * HKV * D * 2);
                }
            } else if (gen9blas_cvt_f16_heads(nv, HKV, D, (const float *) k->data, (long) (k->nb[1] / 4), (long) (k->nb[2] / 4),
                                       (uint16_t *) ks, !span) ||
                gen9blas_cvt_f16_heads(nv, HKV, D, (const float *) v->data, (long) (v->nb[1] / 4), (long) (v->nb[2] / 4),
                                       (uint16_t *) vs, !span)) {
                fprintf(stderr, "gen9: cvt_f16_heads failed: %s\n", gen9blas_error()); fin = false;
            }
            // the kernel takes n_q * G divisible by its RQ rows (4 for G 1): n_q = 265 / 530 / 750 in NeMo's chunks. Main part
            // (n_q rounded down to 4) in place; the last 4 query rows again through a small staging copy, of which the
            // rows past the main part are copied back (no mask, dvis 0: every row sees the same keys, rows independent)
            const int nq4 = nq / 4 * 4, qrow = H * D;
            const size_t kb = (size_t) nk * row + 4096;
            if (fin)
                rc = gen9blas_attn_f16kv(D, H, HKV, nq4, nk, (const float *) q->data, (const uint16_t *) ks, kb,
                                         (const uint16_t *) vs, kb, (float *) dst->data, fa_param(dst, 0), nv, 0, &launches);
            if (fin && !rc && nq4 < nq) {
                static float * qt = nullptr; static float * ot = nullptr;
                if (!qt && (posix_memalign((void **) &qt, 4096, (size_t) 4 * 64 * 1024 * 4) || posix_memalign((void **) &ot, 4096, (size_t) 4 * 64 * 1024 * 4))) qt = ot = nullptr;
                if (!qt || !ot || (size_t) qrow > 64 * 1024) rc = 1;
                else {
                    memcpy(qt, (const float *) q->data + (size_t) (nq - 4) * qrow, (size_t) 4 * qrow * 4);
                    int l2 = 0;
                    rc = gen9blas_attn_f16kv(D, H, HKV, 4, nk, qt, (const uint16_t *) ks, kb, (const uint16_t *) vs, kb, ot,
                                             fa_param(dst, 0), nv, 0, &l2);
                    launches += l2;
                    if (!rc) memcpy((float *) dst->data + (size_t) nq4 * qrow, ot + (size_t) (4 - (nq - nq4)) * qrow,
                                    (size_t) (nq - nq4) * qrow * 4);
                }
            }
        }
    }
    if (rc) {
        if (rc < 0) { n_fa_cpu_err++; fprintf(stderr, "gen9: attention on the GPU failed (%s); ggml-cpu\n", gen9blas_error()); }
        else {
            n_fa_cpu_lib++;
            static std::set<std::pair<int, int>> seen;   // GEN9_INFO: each declined (n_q, n_k) once
            if (getenv("GEN9_INFO") && seen.insert({nq, nv}).second)
                fprintf(stderr, "gen9: F32-K/V attention declined by the library: D %d H %d HKV %d n_q %d n_k %d\n", D, H, HKV, nq, nv);
        }
        fa_on_cpu(ctx, cg, i); return;
    }
    n_fa_gpu++; if (k->type == GGML_TYPE_F16) n_fa_f16gather++; else n_fa_f32kv++; n_fa_launch += launches; fa_flop = fa_flop + 4.0 * H * D * (double) nv * nq;
    t_fa_gpu_s = t_fa_gpu_s + std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
}

static void gen9_flash_attn_streams(gen9_context * ctx, struct ggml_cgraph * cg, int i) {
    struct ggml_tensor * dst = cg->nodes[i];
    const struct ggml_tensor * q = dst->src[0], * k = dst->src[1], * v = dst->src[2], * m = dst->src[3];
    const int D = (int) q->ne[0], nq = (int) q->ne[1], H = (int) q->ne[2], ns = (int) q->ne[3], HKV = (int) k->ne[2], nk = (int) k->ne[1];
    auto t0 = std::chrono::steady_clock::now();
    fa_cpu_inputs(ctx, dst);   // the mask is analysed on the CPU
    std::vector<int> vis(ns);
    int dv = -1;
    for (int st = 0; st < ns; st++) {
        int v0, d;
        const char * mb = (const char *) m->data + (m->ne[3] > 1 ? (size_t) st * m->nb[3] : 0);
        if (!fa_mask_prefix_at(mb, m->nb[1], m->ne[0], nq, v0, d)) { n_fa_streams_cpu++; n_fa_cpu_mask++; fa_on_cpu(ctx, cg, i); return; }
        if (nq == 1) d = 0;
        if (dv < 0) dv = d; else if (dv != d) { n_fa_streams_cpu++; n_fa_cpu_mask++; fa_on_cpu(ctx, cg, i); return; }
        vis[st] = v0;
    }
    int vmax = 0;
    for (int st = 0; st < ns; st++) vmax = std::max(vmax, vis[st] + dv * (nq - 1));
    const int nkl = std::min(nk, (vmax + 15) / 16 * 16);
    auto t1 = std::chrono::steady_clock::now();
    t_fa_mask_s = t_fa_mask_s + std::chrono::duration<double>(t1 - t0).count();
    auto reg_bytes = [](const struct ggml_tensor * t) {
        const struct ggml_tensor * b = t->view_src ? t->view_src : t;
        return (size_t) ((const char *) b->data + ggml_nbytes(b) - (const char *) t->data);
    };
    if (nq <= FA_DEC_MAXQ && fa_dec_ok(dst)) {   // decoding: queued like the elementwise ops (no wait here)
        const int units = ns * nq * HKV;   // work-groups per chunk; aim at ~2 subgroups per hardware thread (24 EU x 7)
        int chunk = (int) ((int64_t) nkl * units / 336 + 15) / 16 * 16;
        chunk = std::max(64, std::min(512, chunk));
        int rc;
        { std::lock_guard<std::mutex> lk(gpu_mu);
          rc = gen9blas_attn_dec(D, H, HKV, ns, nq, nkl, (const float *) q->data, (long) (q->nb[1] / 4), (long) (q->nb[2] / 4), (long) (q->nb[3] / 4),
                                 (const uint16_t *) k->data, (long) (k->nb[1] / 2), (long) (k->nb[2] / 2), (long) (k->nb[3] / 2), reg_bytes(k),
                                 (const uint16_t *) v->data, (long) (v->nb[1] / 2), (long) (v->nb[2] / 2), (long) (v->nb[3] / 2), reg_bytes(v),
                                 (float *) dst->data, fa_param(dst, 0), vis.data(), dv, chunk, ctx->ew_sync); }
        if (rc) { fprintf(stderr, "gen9: decoding attention failed (%s); ggml-cpu\n", gen9blas_error()); n_fa_cpu_err++; n_fa_streams_cpu++; fa_on_cpu(ctx, cg, i); return; }
        ctx->ew_pending = !ctx->ew_sync;
        static const bool verify = getenv("GEN9_ATTN_VERIFY") != nullptr;
        if (verify) {
            ew_flush(ctx);
            const size_t n = ggml_nelements(dst);
            std::vector<float> g((const float *) dst->data, (const float *) dst->data + n);
            fa_on_cpu(ctx, cg, i);
            const float * c = (const float *) dst->data;
            double se = 0, sr = 0;
            for (size_t x = 0; x < n; x++) { const double e = (double) g[x] - c[x]; se += e * e; sr += (double) c[x] * c[x]; }
            const double nmse = sr > 0 ? se / sr : se;
            if (nmse > fa_verify_max) fa_verify_max = nmse;
            n_fa_verify++;
        }
        double fl = 0;
        for (int st = 0; st < ns; st++) for (int t = 0; t < nq; t++) fl += 4.0 * H * D * std::min(vis[st] + dv * t, nkl);
        n_fa_gpu++; n_fa_dec++; n_fa_launch += 2; fa_flop = fa_flop + fl;
        t_fa_gpu_s = t_fa_gpu_s + std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
        return;
    }
    int launches = 0, rc;
    {
        std::lock_guard<std::mutex> lk(gpu_mu);
        rc = gen9blas_attn_f16kv_streams(D, H, HKV, ns, nq, nkl, (const float *) q->data, (long) (q->nb[3] / 4),
                                         (const uint16_t *) k->data, (long) (k->nb[3] / 2), reg_bytes(k), (const uint16_t *) v->data,
                                         (long) (v->nb[3] / 2), reg_bytes(v), (float *) dst->data, fa_param(dst, 0), vis.data(), dv, &launches);
    }
    if (rc) {
        if (rc < 0) { n_fa_cpu_err++; fprintf(stderr, "gen9: stream attention on the GPU failed (%s); ggml-cpu\n", gen9blas_error()); }
        else n_fa_cpu_lib++;
        n_fa_streams_cpu++; fa_on_cpu(ctx, cg, i); return;
    }
    static const bool verify = getenv("GEN9_ATTN_VERIFY") != nullptr;   // diagnostics: GPU result vs ggml-cpu on the same node
    if (verify) {
        const size_t n = ggml_nelements(dst);
        std::vector<float> g((const float *) dst->data, (const float *) dst->data + n);
        fa_on_cpu(ctx, cg, i);
        const float * c = (const float *) dst->data;
        double se = 0, sr = 0;
        for (size_t x = 0; x < n; x++) { const double e = (double) g[x] - c[x]; se += e * e; sr += (double) c[x] * c[x]; }
        const double nmse = sr > 0 ? se / sr : se;
        if (nmse > fa_verify_max) fa_verify_max = nmse;
        n_fa_verify++;
    }
    double fl = 0;
    for (int st = 0; st < ns; st++) for (int t = 0; t < nq; t++) fl += 4.0 * H * D * std::min(vis[st] + dv * t, nkl);
    n_fa_gpu++; n_fa_streams++; n_fa_launch += launches; fa_flop = fa_flop + fl;
    t_fa_gpu_s = t_fa_gpu_s + std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
}

static void gen9_flash_attn(gen9_context * ctx, struct ggml_cgraph * cg, int i) {
    struct ggml_tensor * dst = cg->nodes[i];
    if (dst->src[0]->ne[3] > 1) { gen9_flash_attn_streams(ctx, cg, i); return; }
    if (dst->src[1]->type == GGML_TYPE_F32) { gen9_flash_attn_f32kv(ctx, cg, i); return; }
    const struct ggml_tensor * q = dst->src[0], * k = dst->src[1], * v = dst->src[2], * m = dst->src[3], * rows = dst->src[5];
    const int D = (int) q->ne[0], nq = (int) q->ne[1], H = (int) q->ne[2], HKV = (int) k->ne[2];
    {   // F16 K/V outside the KV-cache layout (accepted in gen9_supports_fa only without mask / kv_rows): staging path
        const size_t row = (size_t) HKV * D * 2;
        if (!m && !rows && (k->nb[1] != row || k->nb[2] != (size_t) D * 2 || v->nb[1] != row || v->nb[2] != (size_t) D * 2)) {
            gen9_flash_attn_f32kv(ctx, cg, i); return;
        }
    }
    int vis0 = (int) k->ne[1], dvis = 0;
    fa_cpu_inputs(ctx, dst);   // mask and kv_rows are read on the CPU
    auto t0 = std::chrono::steady_clock::now();
    if (m && !fa_mask_prefix(m, nq, vis0, dvis)) {
        std::vector<fa_run> runs;
        const bool ok = fa_mask_runs(m, nq, runs) && runs.size() * 8 <= (size_t) nq;
        t_fa_mask_s = t_fa_mask_s + std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (!ok) { n_fa_cpu_mask++; fa_on_cpu(ctx, cg, i); return; }
        gen9_flash_attn_runs(ctx, cg, i, runs);
        return;
    }
    const int64_t vmax = std::min<int64_t>(vis0 + (int64_t) dvis * (nq - 1), rows ? rows->ne[0] : k->ne[1]);
    if (rows) {   // kv_rows: column c reads cache row rows[c]; identity over the visible columns = a plain view
        const int32_t * r = (const int32_t *) rows->data;
        for (int64_t c = 0; c < vmax; c++) if (r[c] != c) {   // not the identity: gather through the general path
            std::vector<fa_run> runs;
            if (m && fa_mask_runs(m, nq, runs) && runs.size() * 8 <= (size_t) nq) { gen9_flash_attn_runs(ctx, cg, i, runs); return; }
            n_fa_cpu_rows++; fa_on_cpu(ctx, cg, i); return;
        }
        n_fa_rows++;
    }
    auto t1 = std::chrono::steady_clock::now();
    t_fa_mask_s = t_fa_mask_s + std::chrono::duration<double>(t1 - t0).count();
    int nk = (int) std::min<int64_t>((vmax + 15) / 16 * 16, k->ne[1]);   // key rows the kernel reads (multiple of 16)
    if (nk % 16) {   // n_k itself not a multiple of 16 (no mask, checked in supports_op): read the rows behind the view
        nk = (nk + 15) / 16 * 16;
        ew_flush(ctx);     // the V rows below are read on the CPU (queued SET_ROWS first)
        bool fin = true;   // keys >= vis0 get p = 0, but 0 * Inf/NaN in V is NaN: their V rows must be finite
        for (int64_t j = vmax; j < nk && fin; j++) {
            const uint16_t * r = (const uint16_t *) ((const char *) v->data + j * v->nb[1]);
            for (int64_t x = 0; x < (int64_t) HKV * D; x++) if ((r[x] & 0x7C00) == 0x7C00) { fin = false; break; }   // exponent all ones
        }
        if (!fin) { n_fa_cpu_tail++; fa_on_cpu(ctx, cg, i); return; }
        n_fa_tail++;
    }
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
        if (rc < 0) { n_fa_cpu_err++; fprintf(stderr, "gen9: attention on the GPU failed (%s); ggml-cpu\n", gen9blas_error()); }
        else n_fa_cpu_lib++;
        fa_on_cpu(ctx, cg, i); return;
    }
    double fl = 0;
    for (int t = 0; t < nq; t++) fl += 4.0 * H * D * std::min<int64_t>(vis0 + (int64_t) dvis * t, nk);
    n_fa_gpu++; n_fa_launch += launches; fa_flop = fa_flop + fl;
    t_fa_gpu_s = t_fa_gpu_s + std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
}

// ---------------- MUL_MAT groups on one input (Q/K/V) ----------------
// At a weight MUL_MAT, later weight MUL_MATs of the same split with the same input (Q, K, V after the attention norm) are
// computed with it: input packed once, one wait (gen9blas_gemm_multi_f16w). Computing a later node early is only valid
// if its output memory is not used by any node in between (ggml-alloc may give it memory of a tensor that dies between
// the two nodes): every candidate's byte range is checked against the tensors (sources and results) of the nodes in
// between, otherwise the candidate is left for its turn. GEN9_QKV=0 turns it off.
static bool multi_off(void) { static const bool v = [] { const char * e = getenv("GEN9_QKV"); return e && !strcmp(e, "0"); }(); return v; }
static bool overlaps(const struct ggml_tensor * a, const struct ggml_tensor * b) {
    if (!a || !b || !a->data || !b->data) return false;
    const char * a0 = (const char *) a->data, * b0 = (const char *) b->data;
    return a0 < b0 + ggml_nbytes(b) && b0 < a0 + ggml_nbytes(a);
}
static bool mm_f32_forced(void) { static const bool v = [] { const char * e = getenv("GEN9_MM_F32"); return e && !strcmp(e, "1"); }(); return v; }
static void mm_f32_recompute(struct ggml_tensor * dst) {   // F32 weights for one MUL_MAT after FP16 overflow
    const struct ggml_tensor * w = dst->src[0], * x = dst->src[1];
    const int K = (int) w->ne[0], N = (int) w->ne[1], M = (int) x->ne[1];
    std::vector<float> wf((size_t) N * K);
    dequant_rows(w, wf.data(), false);
    std::lock_guard<std::mutex> lk(gpu_mu);
    if (gen9blas_gemm(GEN9BLAS_S, 0, 1, M, N, K, 1.0, x->data, (int) (x->nb[1] / 4), wf.data(), K, 0.0, dst->data, (int) (dst->nb[1] / 4)))
        fprintf(stderr, "gen9: F32 recompute of %s failed: %s\n", dst->name, gen9blas_error());
}
static bool gen9_try_multi(gen9_context * ctx, struct ggml_cgraph * cg, int i) {
    struct ggml_tensor * n0 = cg->nodes[i];
    if (multi_off() || !is_f16_weight_mm(n0)) return false;
    const struct ggml_tensor * x = n0->src[1];
    if (x->type != GGML_TYPE_F32 || !ggml_is_contiguous(x) || x->ne[2] != 1 || x->ne[3] != 1) return false;
    std::vector<int> grp{ i };
    for (int j = i + 1; j < cg->n_nodes && j <= i + 24 && grp.size() < 4; j++) {
        struct ggml_tensor * c = cg->nodes[j];
        if (!(c->flags & GGML_TENSOR_FLAG_COMPUTE) || c->src[1] != x || !is_f16_weight_mm(c) || c->src[0]->ne[0] != n0->src[0]->ne[0] ||
            !ggml_is_contiguous(c) || c->ne[1] != n0->ne[1]) continue;
        bool clash = overlaps(c, x);
        for (int k = i; k < j && !clash; k++) {
            const struct ggml_tensor * m = cg->nodes[k];
            clash = overlaps(c, m);
            for (int s = 0; s < GGML_MAX_SRC && !clash; s++) clash = overlaps(c, m->src[s]);
        }
        if (clash) { n_multi_overlap++; continue; }
        grp.push_back(j);
    }
    if (grp.size() < 2) return false;
    const int n = (int) grp.size(), M = (int) x->ne[1];
    std::vector<packed_t> P(n);
    int rc;
    std::vector<int> nf(n), ldc(n);
    {
        std::lock_guard<std::mutex> lk(gpu_mu);
        bool ok = true;
        for (int g = 0; g < n; g++) { P[g] = get_packed(ctx, cg->nodes[grp[g]]->src[0]); ok &= P[g].P && P[g].t == GEN9BLAS_HS; }
        std::vector<const gen9blas_packed_b *> pp(n); std::vector<float *> C(n);
        for (int g = 0; g < n; g++) { pp[g] = P[g].P; C[g] = (float *) cg->nodes[grp[g]]->data; ldc[g] = (int) (cg->nodes[grp[g]]->nb[1] / 4); }
        rc = ok ? gen9blas_gemm_multi_f16w(M, (const float *) x->data, (int) (x->nb[1] / 4), n, pp.data(), C.data(), ldc.data(), nf.data()) : 1;
        for (int g = 0; g < n; g++) release(P[g]);
    }
    if (rc < 0) { fprintf(stderr, "gen9: gemm_multi_f16w failed: %s\n", gen9blas_error()); return false; }
    if (rc > 0) return false;
    if (!ctx->nw) ctx->ew_pending = false;   // the call waited for everything queued before it (not in no-wait mode)
    for (int g = 0; g < n; g++) {
        if (nf[g]) { n_multi_fallback++; mm_f32_recompute(cg->nodes[grp[g]]); }
        if (g) ctx->done[grp[g]] = 1;
    }
    n_multi_groups++; n_multi_mm += n;
    return true;
}

// ---------------- per-layer elementwise ops: RMS_NORM (+ its weight MUL), ADD, MUL, ROPE ----------------
// They only enqueue: the kernels run in the batch of the next GEMM / attention call, which waits anyway, or at the end
// of the split (gen9_graph_compute). The ggml buffers written in the split are registered as kcl spans first, so a tensor
// written by one queued op and read by the next has one GPU address (docs/no-neo.md 6g.5); if a buffer cannot be
// registered, every op of the split waits (counted). Anything in this backend that reads results on the CPU flushes first.
// GEN9_EWOPS=0 turns them off (supports_op declines), GEN9_EW_SYNC=1 forces the wait after each op (diagnostics).
static bool ew_off(void) { static const bool v = [] { const char * e = getenv("GEN9_EWOPS"); return e && !strcmp(e, "0"); }(); return v; }
static bool ew_force_sync(void) { static const bool v = getenv("GEN9_EW_SYNC") != nullptr; return v; }
static bool is_ew_op(enum ggml_op o) { return o == GGML_OP_RMS_NORM || o == GGML_OP_NORM || o == GGML_OP_ADD || o == GGML_OP_MUL || o == GGML_OP_ROPE || o == GGML_OP_UNARY || o == GGML_OP_IM2COL || o == GGML_OP_SET_ROWS ||
                                        o == GGML_OP_CPY || o == GGML_OP_CONT || o == GGML_OP_DUP || o == GGML_OP_UPSCALE; }

static void ew_flush(gen9_context * ctx) {
    if (!ctx->ew_pending) return;
    std::lock_guard<std::mutex> lk(gpu_mu);
    if (gen9blas_flush()) fprintf(stderr, "gen9: flush failed: %s\n", gen9blas_error());
    ctx->ew_pending = false;
}
static bool ew_after(gen9_context * ctx, int rc, const char * what) {
    if (rc) { fprintf(stderr, "gen9: %s failed: %s\n", what, gen9blas_error()); return false; }
    ctx->ew_pending = !ctx->ew_sync;
    return true;
}

// RMS_NORM at node i, fused with the next node when that is MUL(norm, weight row) and the norm has no other use.
// Returns the number of nodes consumed (1 or 2), 0 on failure.
static int gen9_rms_norm(gen9_context * ctx, struct ggml_cgraph * cg, int i) {
    struct ggml_tensor * node = cg->nodes[i];
    const struct ggml_tensor * x = node->src[0];
    float eps; memcpy(&eps, node->op_params, sizeof eps);
    const int K = (int) x->ne[0], M = (int) ggml_nrows(x);
    struct ggml_tensor * out = node; const float * w = nullptr;
    if (i + 1 < cg->n_nodes) {
        struct ggml_tensor * mul = cg->nodes[i + 1];
        const struct ggml_tensor * wt = mul->src[0] == node ? mul->src[1] : mul->src[0];
        if (mul->op == GGML_OP_MUL && (mul->flags & GGML_TENSOR_FLAG_COMPUTE) && (mul->src[0] == node) != (mul->src[1] == node) &&
            wt->type == GGML_TYPE_F32 && ggml_is_contiguous(wt) && ggml_nrows(wt) == 1 && wt->ne[0] == K &&
            ggml_are_same_shape(mul, node) && ggml_is_contiguous(mul) && ggml_node_has_n_uses(cg, i, 1)) {
            out = mul; w = (const float *) wt->data;
        }
    }
    int rc;
    { std::lock_guard<std::mutex> lk(gpu_mu);
      rc = gen9blas_rms_norm(M, K, (const float *) x->data, K, w, eps, (float *) out->data, K, ctx->ew_sync); }
    if (!ew_after(ctx, rc, "RMS_NORM")) return 0;
    n_ew_rms++;
    if (w) { n_ew_rms_mul++; return 2; }
    return 1;
}

// NORM at node i, fused with MUL(norm, w row) and then ADD(that, b row) when they follow and have no other use.
static bool normpack_off(void) { static const bool v = [] { const char * e = getenv("GEN9_NORMPACK"); return e && !strcmp(e, "0"); }(); return v; }
// does a later MUL_MAT with an FP16-packed weight of width K read `out`?
static bool norm_feeds_mm(struct ggml_cgraph * cg, int from, const struct ggml_tensor * out, int K) {
    for (int j = from; j < cg->n_nodes; j++) {
        const struct ggml_tensor * n = cg->nodes[j];
        if (n->op == GGML_OP_MUL_MAT && n->src[1] == out && ffn_weight_type(n->src[0]->type) && is_weight(n->src[0]) &&
            n->src[0]->ne[0] == K && n->type == GGML_TYPE_F32) return true;
    }
    return false;
}
static int gen9_layer_norm(gen9_context * ctx, struct ggml_cgraph * cg, int i) {
    struct ggml_tensor * node = cg->nodes[i];
    const struct ggml_tensor * x = node->src[0];
    float eps; memcpy(&eps, node->op_params, sizeof eps);
    const int K = (int) x->ne[0], M = (int) ggml_nrows(x);
    auto row_operand = [&](struct ggml_tensor * n, const struct ggml_tensor * prev) -> const struct ggml_tensor * {
        if (!(n->flags & GGML_TENSOR_FLAG_COMPUTE) || (n->src[0] == prev) == (n->src[1] == prev)) return nullptr;
        const struct ggml_tensor * o = n->src[0] == prev ? n->src[1] : n->src[0];
        return o->type == GGML_TYPE_F32 && ggml_is_contiguous(o) && ggml_nrows(o) == 1 && o->ne[0] == K &&
               ggml_are_same_shape(n, prev) && ggml_is_contiguous(n) ? o : nullptr;
    };
    struct ggml_tensor * out = node; const float * w = nullptr, * b = nullptr; int used = 1;
    if (i + 1 < cg->n_nodes && cg->nodes[i + 1]->op == GGML_OP_MUL && ggml_node_has_n_uses(cg, i, 1)) {
        if (const struct ggml_tensor * wt = row_operand(cg->nodes[i + 1], node)) {
            out = cg->nodes[i + 1]; w = (const float *) wt->data; used = 2;
            if (i + 2 < cg->n_nodes && cg->nodes[i + 2]->op == GGML_OP_ADD && ggml_node_has_n_uses(cg, i + 1, 1))
                if (const struct ggml_tensor * bt = row_operand(cg->nodes[i + 2], out)) { out = cg->nodes[i + 2]; b = (const float *) bt->data; used = 3; }
        }
    }
    // the GEMM input packed by the norm itself when a MUL_MAT with an FP16-packed weight reads this output (whisper:
    // Q, K, V after norm1, fc1 after norm2); GEN9_NORMPACK=0 turns it off
    gen9blas_packed_a * pa = nullptr;
    if (!normpack_off() && gen9blas_has(GEN9BLAS_FEAT_NORM_PACK) && norm_feeds_mm(cg, i + used, out, K)) {
        const int s = ctx->pa_next++ & 1;
        if (!ctx->pa_pool[s]) ctx->pa_pool[s] = gen9blas_packed_a_create();
        pa = ctx->pa_pool[s];
        for (auto it = ctx->pa_map.begin(); it != ctx->pa_map.end();) it = it->second.pa == pa ? ctx->pa_map.erase(it) : std::next(it);
    }
    int rc;
    { std::lock_guard<std::mutex> lk(gpu_mu);
      rc = pa ? gen9blas_layer_norm_pack(M, K, (const float *) x->data, K, w, b, eps, (float *) out->data, K, pa, ctx->ew_sync)
              : gen9blas_layer_norm(M, K, (const float *) x->data, K, w, b, eps, (float *) out->data, K, ctx->ew_sync); }
    if (!ew_after(ctx, rc, "NORM")) return 0;
    if (pa) { ctx->pa_map[out] = { pa, out->data }; n_normpack++; }
    else ctx->pa_map.erase(out);
    n_ew_norm++;
    if (used == 3) n_ew_norm_wb++;
    return used;
}

// ggml IM2COL (conv_1d / conv_2d front end) as ggml-cpu: F32 input with contiguous rows, F32 or F16 output.
static int ew_min_rows(void);
static bool im2col_ok(const struct ggml_tensor * op) {
    const struct ggml_tensor * k = op->src[0], * x = op->src[1];
    const bool is2 = ggml_get_op_params_i32(op, 6) == 1;
    if (!k || !x || x->type != GGML_TYPE_F32 || (op->type != GGML_TYPE_F32 && op->type != GGML_TYPE_F16) || !ggml_is_contiguous(op) ||
        x->nb[0] != 4 || (is2 && x->nb[1] != (size_t) x->ne[0] * 4)) return false;
    const int64_t N = is2 ? x->ne[3] : x->ne[2], IC = is2 ? x->ne[2] : x->ne[1], KK = (is2 ? k->ne[1] : 1) * k->ne[0];
    const int64_t OH = is2 ? op->ne[2] : 1, OW = op->ne[1];
    return (is2 || x->ne[3] == 1) && IC * KK <= INT_MAX && N * OH <= INT_MAX && OW <= INT_MAX && (size_t) (is2 ? x->nb[3] : x->nb[2]) % 4 == 0 &&
           (size_t) (is2 ? x->nb[2] : x->nb[1]) % 4 == 0 && N * OH * OW >= ew_min_rows();
}
static bool gen9_im2col(gen9_context * ctx, struct ggml_tensor * node) {
    const struct ggml_tensor * k = node->src[0], * x = node->src[1];
    const int32_t * pp = (const int32_t *) node->op_params;
    const bool is2 = pp[6] == 1;
    const int N = (int) (is2 ? x->ne[3] : x->ne[2]), IC = (int) (is2 ? x->ne[2] : x->ne[1]), IH = is2 ? (int) x->ne[1] : 1, IW = (int) x->ne[0];
    const int KH = is2 ? (int) k->ne[1] : 1, KW = (int) k->ne[0], OH = is2 ? (int) node->ne[2] : 1, OW = (int) node->ne[1];
    const long ofs0 = (long) ((is2 ? x->nb[3] : x->nb[2]) / 4), ofs1 = (long) ((is2 ? x->nb[2] : x->nb[1]) / 4);
    int rc;
    { std::lock_guard<std::mutex> lk(gpu_mu);
      rc = gen9blas_im2col(N, IC, IH, IW, (const float *) x->data, ofs0, ofs1, KH, KW, OH, OW, pp[0], pp[1], pp[2], pp[3], pp[4], pp[5],
                           node->data, node->type == GGML_TYPE_F16, ctx->ew_sync); }
    if (!ew_after(ctx, rc, "IM2COL")) return false;
    n_ew_im2col++;
    return true;
}

static bool gen9_gelu(gen9_context * ctx, struct ggml_tensor * node) {
    int rc;
    { std::lock_guard<std::mutex> lk(gpu_mu);
      rc = ggml_get_unary_op(node) == GGML_UNARY_OP_RELU
          ? gen9blas_relu((size_t) ggml_nelements(node), (const float *) node->src[0]->data, (float *) node->data, ctx->ew_sync)
          : ggml_get_unary_op(node) == GGML_UNARY_OP_GELU_ERF
          ? gen9blas_gelu_erf((size_t) ggml_nelements(node), (const float *) node->src[0]->data, (float *) node->data, ctx->ew_sync)
          : gen9blas_gelu((size_t) ggml_nelements(node), (const float *) node->src[0]->data, (float *) node->data, ctx->ew_sync); }
    if (!ew_after(ctx, rc, "GELU")) return false;
    n_ew_gelu++;
    return true;
}

static bool gen9_binop(gen9_context * ctx, struct ggml_tensor * node) {
    const struct ggml_tensor * a = node->src[0], * b = node->src[1];
    const int K = (int) a->ne[0], M = (int) ggml_nrows(a);
    const bool col = b->ne[0] == 1 && K > 1;   // one value per row (conv bias [1, rows])
    const int ldb = col ? 1 : ggml_nrows(b) == 1 ? 0 : K, bs0 = col ? 0 : 1;
    const int op = node->op == GGML_OP_MUL;
    int rc;
    { std::lock_guard<std::mutex> lk(gpu_mu);
      rc = gen9blas_binop_rows(op, M, K, (const float *) a->data, K, (const float *) b->data, ldb, bs0, (float *) node->data, K, ctx->ew_sync); }
    if (!ew_after(ctx, rc, op ? "MUL" : "ADD")) return false;
    (op ? n_ew_mul : n_ew_add)++;
    return true;
}

static bool gen9_rope(gen9_context * ctx, struct ggml_tensor * node) {
    const struct ggml_tensor * x = node->src[0], * pos = node->src[1], * ff = node->src[2];
    const int32_t * pp = (const int32_t *) node->op_params;
    const int n_dims = pp[1], mode = pp[2], n_ctx_orig = pp[4];
    float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow, corr[2];
    memcpy(&freq_base, pp + 5, 4); memcpy(&freq_scale, pp + 6, 4); memcpy(&ext_factor, pp + 7, 4);
    memcpy(&attn_factor, pp + 8, 4); memcpy(&beta_fast, pp + 9, 4); memcpy(&beta_slow, pp + 10, 4);
#ifndef GEN9_GGML_NO_ROPE_TRUNCATE
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, ggml_rope_get_truncate(node), corr);
#else   // older ggml (whisper.cpp): no truncate; ROPE is declined in supports_op, this is never reached
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr);
#endif
    int rc;
    if (mode == GGML_ROPE_TYPE_MROPE || mode == GGML_ROPE_TYPE_VISION) {   // ggml_rope_multi: sections at op_params[11..14], 4 positions per token
        int sec[4]; memcpy(sec, pp + 11, sizeof sec);
        const int vis = mode == GGML_ROPE_TYPE_VISION;
        std::lock_guard<std::mutex> lk(gpu_mu);
        rc = gen9blas_rope_multi(n_dims, (int) x->ne[1], (int) x->ne[2], (const float *) x->data, (long) (x->nb[1] / 4),
                                 (long) (x->nb[2] / 4), (float *) node->data, (long) (node->nb[1] / 4), (long) (node->nb[2] / 4),
                                 (const int *) pos->data, ff ? (const float *) ff->data : nullptr, sec, freq_base, freq_scale,
                                 ext_factor, attn_factor, corr, vis, ctx->ew_sync);
        if (!ew_after(ctx, rc, "ROPE")) return false;
        n_ew_rope++; if (vis) n_ew_vrope++;
        return true;
    }
    { std::lock_guard<std::mutex> lk(gpu_mu);
      rc = gen9blas_rope(n_dims, (int) x->ne[1], (int) x->ne[2], (const float *) x->data, (long) (x->nb[1] / 4), (long) (x->nb[2] / 4),
                         (float *) node->data, (long) (node->nb[1] / 4), (long) (node->nb[2] / 4), (const int *) pos->data,
                         ff ? (const float *) ff->data : nullptr, mode == GGML_ROPE_TYPE_NEOX, freq_base, freq_scale, ext_factor,
                         attn_factor, corr, ctx->ew_sync); }
    if (!ew_after(ctx, rc, "ROPE")) return false;
    n_ew_rope++;
    return true;
}

// CPY / CONT / DUP between F32 and F16, any strides (gen9blas_cpy_any). The kernel reads and writes whole elements, so
// the data pointers must be aligned to the element size (always the case for ggml buffers and views of whole elements);
// otherwise ggml-cpu computes the node (counted).
static bool gen9_cpy(gen9_context * ctx, struct ggml_cgraph * cg, int i) {
    struct ggml_tensor * node = cg->nodes[i];
    const struct ggml_tensor * a = node->src[0];
    const size_t ea = ggml_type_size(a->type), ed = ggml_type_size(node->type);
    if ((uintptr_t) a->data % ea || (uintptr_t) node->data % ed) { cpu_node(ctx, cg, i); n_ew_cpy_cpu++; return true; }
    long sne[4], snb[4], dne[4], dnb[4];
    gen9_shape(a, sne, snb); gen9_shape(node, dne, dnb);
    int rc;
    { std::lock_guard<std::mutex> lk(gpu_mu);
      rc = gen9blas_cpy_any(a->data, a->type == GGML_TYPE_F16 ? GEN9BLAS_H : GEN9BLAS_S, sne, snb, node->data, node->type == GGML_TYPE_F16 ? GEN9BLAS_H : GEN9BLAS_S, dne, dnb, ctx->ew_sync); }
    if (!ew_after(ctx, rc, "CPY")) return false;
    n_ew_cpy++;
    return true;
}
static bool gen9_upscale(gen9_context * ctx, struct ggml_tensor * node) {
    const struct ggml_tensor * a = node->src[0];
    long sne[4], snb[4], dne[4], dnb[4];
    gen9_shape(a, sne, snb); gen9_shape(node, dne, dnb);
    int rc;
    { std::lock_guard<std::mutex> lk(gpu_mu);
      rc = gen9blas_upscale_f32((const float *) a->data, sne, snb, (float *) node->data, dne, dnb, ggml_get_op_params_i32(node, 0), ctx->ew_sync); }
    if (!ew_after(ctx, rc, "UPSCALE")) return false;
    n_ew_upscale++;
    return true;
}

static bool gen9_set_rows(gen9_context * ctx, struct ggml_tensor * node) {
    const struct ggml_tensor * a = node->src[0], * ix = node->src[1];
    const size_t ext = (size_t) (node->ne[1] - 1) * node->nb[1] + (size_t) (node->ne[2] - 1) * node->nb[2] +
                       (size_t) (node->ne[3] - 1) * node->nb[3] + (size_t) node->ne[0] * 2;
    const size_t iext = (size_t) (ix->ne[0] - 1) * ix->nb[0] + (size_t) (ix->ne[1] - 1) * ix->nb[1] +
                        (size_t) (ix->ne[2] - 1) * ix->nb[2] + ggml_type_size(ix->type);
    int rc, bad = 0;
    { std::lock_guard<std::mutex> lk(gpu_mu);
      rc = gen9blas_set_rows_f16((int) a->ne[0], (int) a->ne[1], (int) a->ne[2], (int) a->ne[3], (const float *) a->data,
                                 (long) (a->nb[1] / 4), (long) (a->nb[2] / 4), (long) (a->nb[3] / 4), ix->data, iext,
                                 (long) ix->nb[0], (long) ix->nb[1], (long) ix->nb[2], ix->type == GGML_TYPE_I64,
                                 (int) ix->ne[1], (int) ix->ne[2], (uint16_t *) node->data, ext, (long) (node->nb[1] / 2),
                                 (long) (node->nb[2] / 2), (long) (node->nb[3] / 2), (long) node->ne[1], ctx->ew_sync, &bad); }
    if (bad) { fprintf(stderr, "gen9: SET_ROWS index out of range\n"); return false; }
    if (!ew_after(ctx, rc, "SET_ROWS")) return false;
    n_ew_setrows++;
    return true;
}

// Start of a split: when it holds elementwise ops, register the buffers its nodes write as spans (else: wait per op).
static void ew_begin(gen9_context * ctx, struct ggml_cgraph * cg) {
    ctx->ew_sync = ew_force_sync() || prof_on(); ctx->ew_pending = false; ctx->ew_cpu = false;
    bool any = false, waits = false;
    std::set<ggml_backend_buffer_t> bufs;
    for (int i = 0; i < cg->n_nodes; i++) {
        const struct ggml_tensor * n = cg->nodes[i];
        if (!(n->flags & GGML_TENSOR_FLAG_COMPUTE)) continue;
        // stream attention may run as the queued decoding kernel: its O must live in a span like an elementwise result
        any |= is_ew_op(n->op) || (n->op == GGML_OP_FLASH_ATTN_EXT && n->src[0]->ne[3] > 1);
        waits |= n->op == GGML_OP_MUL_MAT || n->op == GGML_OP_FLASH_ATTN_EXT;
        // only nodes that write: a RESHAPE/VIEW/PERMUTE/TRANSPOSE/NONE node of a weight lives in the weights' buffer,
        // which NeMo-Speech.cpp maps read-only from the GGUF without marking it WEIGHTS -> a writable span on it made
        // every execbuffer fail with EFAULT ("Bad address")
        if (ggml_op_is_empty(n->op)) continue;
        ggml_backend_buffer_t b = n->view_src ? n->view_src->buffer : n->buffer;
        if (b && b->usage != GGML_BACKEND_BUFFER_USAGE_WEIGHTS) bufs.insert(b);
    }
    // No call in this split waits for the GPU anyway: queued ops would cost a wait of their own at the end of the split
    // (~0.2 ms), more than ggml-cpu needs for them at these sizes (whisper's decoder prompt was 6 % slower).
    if (any && !waits && !getenv("GEN9_EW_LONE_GPU")) { ctx->ew_cpu = true; n_ew_cpu_splits++; return; }
    if ((!any && !ctx->nw) || ctx->ew_sync) { if (ctx->ew_sync) ctx->nw = false; return; }
    std::lock_guard<std::mutex> lk(gpu_mu);
    for (ggml_backend_buffer_t b : bufs)
        if (gen9blas_span(ggml_backend_buffer_get_base(b), ggml_backend_buffer_get_size(b), 1)) { ctx->ew_sync = true; break; }
    if (ctx->ew_sync) { gen9blas_span_clear(); n_ew_sync_graphs++; ctx->nw = false; }
}
static void ew_end(gen9_context * ctx) {
    ew_flush(ctx);
    std::lock_guard<std::mutex> lk(gpu_mu);
    gen9blas_span_clear();
}

// Whisper encoder: MUL_MAT (F16 or quantized weight, F32 activations) -> ADD(row bias) [-> GELU | -> ADD(residual)], each
// intermediate used once: one GEMM whose epilogue adds the bias, applies GELU and adds the residual
// (gen9blas_gemm_pb_f16w_ex, bit-identical to the separate kernels; results/whisper-epilog-fusion-model-2026-11-14.txt).
// Returns the number of nodes computed (0 = not fused). GEN9_EPI=0 turns it off.
static bool epi_off(void) { static const bool v = [] { const char * e = getenv("GEN9_EPI"); return e && !strcmp(e, "0"); }(); return v; }
static bool rows_f32(const struct ggml_tensor * t) {   // 2-D F32 with contiguous rows
    return t->type == GGML_TYPE_F32 && t->ne[2] == 1 && t->ne[3] == 1 && t->nb[0] == sizeof(float);
}
static bool overlap(const struct ggml_tensor * a, const struct ggml_tensor * b) {
    const char * a0 = (const char *) a->data, * b0 = (const char *) b->data;
    return a0 < b0 + ggml_nbytes(b) && b0 < a0 + ggml_nbytes(a);
}
static int gen9_try_epi(gen9_context * ctx, struct ggml_cgraph * cg, int i) {
    if (epi_off() || !gen9blas_has(GEN9BLAS_FEAT_EPILOG_EXT)) return 0;
    struct ggml_tensor * mm = cg->nodes[i];
    const struct ggml_tensor * w = mm->src[0], * x = mm->src[1];
    // F16 or a quantized weight with to_float (Q8_0 model): both are packed as FP16 (HS), the epilogue is the same
    if (!ffn_weight_type(w->type) || x->type != GGML_TYPE_F32 || !is_weight(w) || !rows_f32(mm) || !rows_f32(x)) return 0;
    if (i + 1 >= cg->n_nodes || !ggml_node_has_n_uses(cg, i, 1)) return 0;
    struct ggml_tensor * add = cg->nodes[i + 1];
    if (add->op != GGML_OP_ADD || add->src[0] != mm || !ggml_are_same_shape(add, mm) || !rows_f32(add)) return 0;
    const struct ggml_tensor * bias = add->src[1];
    const int N = (int) mm->ne[0], M = (int) mm->ne[1], K = (int) w->ne[0];
    if (!rows_f32(bias) || bias->ne[0] != N || bias->ne[1] != 1 || overlap(bias, add)) return 0;
    struct ggml_tensor * out = add;
    int gelu = 0, used = 2;
    const float * R = nullptr; int ldr = 0;
    if (i + 2 < cg->n_nodes && ggml_node_has_n_uses(cg, i + 1, 1)) {
        struct ggml_tensor * nx = cg->nodes[i + 2];
        if (nx->op == GGML_OP_UNARY && ggml_get_unary_op(nx) == GGML_UNARY_OP_GELU && nx->src[0] == add &&
            ggml_are_same_shape(nx, add) && rows_f32(nx)) {
            gelu = 1; out = nx; used = 3;
        } else if (nx->op == GGML_OP_ADD && (nx->src[0] == add || nx->src[1] == add) && ggml_are_same_shape(nx, add) && rows_f32(nx)) {
            const struct ggml_tensor * r = nx->src[0] == add ? nx->src[1] : nx->src[0];
            if (ggml_are_same_shape(r, add) && rows_f32(r)) {
                if (overlap(r, nx)) n_epi_res_alias++;   // in place: a fused write would destroy R before a possible
                else { R = (const float *) r->data; ldr = (int) (r->nb[1] / sizeof(float)); out = nx; used = 3; }   // F32 redo
            }
        }
    }
    int rc, nonfinite = 0;
    {
        std::lock_guard<std::mutex> lk(gpu_mu);
        packed_t p = get_packed(ctx, w);
        if (!p.P || p.t != GEN9BLAS_HS) { if (p.P && p.temp) gen9blas_packed_b_free(p.P); return 0; }
        auto pit = ctx->pa_map.find(x);
        if (pit != ctx->pa_map.end() && pit->second.data == x->data) {
            n_pa_used++;
            rc = gen9blas_gemm_pa_f16w_ex(pit->second.pa, p.P, (float *) out->data, (int) (out->nb[1] / sizeof(float)),
                                          (const float *) bias->data, gelu, R, ldr, &nonfinite);
        } else
            rc = gen9blas_gemm_pb_f16w_ex(M, (const float *) x->data, (int) (x->nb[1] / sizeof(float)), p.P, (float *) out->data,
                                          (int) (out->nb[1] / sizeof(float)), (const float *) bias->data, gelu, R, ldr, &nonfinite);
        if (p.temp) { gen9blas_flush(); gen9blas_packed_b_free(p.P); }
    }
    (void) K;
    if (rc) { fprintf(stderr, "gen9: fused MUL_MAT %s failed: %s\n", w->name, gen9blas_error()); return 0; }
    if (nonfinite) {   // FP16 overflow: the MUL_MAT alone the usual way (it redoes it in F32), then the next nodes one by one
        n_epi_fallback++;
        return gen9_mul_mat(ctx, mm) ? 1 : -1;
    }
    n_epi++; n_mm_f16++; if (gelu) n_epi_gelu++; if (R) n_epi_res++;
    return used;
}

static enum ggml_status gen9_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    gen9_context * ctx = (gen9_context *) backend->context;
    n_graphs++;
    const auto pg0 = std::chrono::steady_clock::now();
    struct pr_guard { std::chrono::steady_clock::time_point t; ~pr_guard() { if (prof_on()) pr_graph_s = pr_graph_s + std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count(); } } pguard{ pg0 };
    // GEN9_MM_NOWAIT=1: a split whose products all have M <= 16 (several sequences decoded together) queues them without
    // waiting; one wait at the end, then the FP16-overflow flags: any overflow fails the graph (the caller retries without)
    ctx->nw = false;
    if (mm_nowait_on()) {
        bool mm = false, small = true;
        for (int j = 0; j < cgraph->n_nodes; j++) {
            const struct ggml_tensor * n = cgraph->nodes[j];
            if (n->op != GGML_OP_MUL_MAT || !(n->flags & GGML_TENSOR_FLAG_COMPUTE)) continue;
            mm = true; small &= n->src[1]->ne[1] <= 16;
        }
        ctx->nw = mm && small;
    }
    ew_begin(ctx, cgraph);
    struct ew_guard { gen9_context * c; ~ew_guard() { ew_end(c); } } guard{ ctx };
    struct nw_guard { gen9_context * c; ~nw_guard() { if (c->nw) { std::lock_guard<std::mutex> lk(gpu_mu); gen9blas_flush(); gen9blas_nowait_pending(); gen9blas_set_nowait(0); c->nw = false; } } } nwg{ ctx };
    if (ctx->nw) { std::lock_guard<std::mutex> lk(gpu_mu); gen9blas_set_nowait(1); n_nw_splits++; }
    ctx->done.assign(cgraph->n_nodes, 0);
    // pa_map is NOT cleared per split: whisper's norm1 output feeds K, V and Q in three different splits (the K / V
    // copies into the cache run on the CPU in between); entries are checked against the tensor's data pointer
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];
        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) { n_noop++; continue; }
        if (ctx->done[i]) continue;
        const bool prof = prof_on(); const int i0 = i; int pk = -1;
        std::chrono::steady_clock::time_point pt0;
        std::vector<char> done0;
        if (prof) { pt0 = std::chrono::steady_clock::now(); if (node->op == GGML_OP_MUL_MAT) done0 = ctx->done; }
        switch (node->op) {
            case GGML_OP_NONE: case GGML_OP_RESHAPE: case GGML_OP_VIEW: case GGML_OP_PERMUTE: case GGML_OP_TRANSPOSE:
                n_noop++;
                break;
            case GGML_OP_MUL_MAT:
                // GEN9_MM_F32=1 (diagnostics): weight MUL_MATs with F32 activations in full F32 (weights dequantized to F32
                // per call, SGEMM) instead of the FP16 paths -- to measure what FP16 costs in accuracy; slow
                if (mm_f32_forced() && node->src[1]->type == GGML_TYPE_F32 && node->src[0]->type != GGML_TYPE_F32 &&
                    ggml_is_contiguous(node->src[1]) && node->src[1]->ne[2] == 1 && node->src[1]->ne[3] == 1) {
                    ew_flush(ctx); mm_f32_recompute(node); n_mm_forced_f32++; pk = PR_MM; break;
                }
                if (ctx->nw) ctx->ew_pending = true;   // no-wait products: a CPU node or the split end must flush first
                if (gen9_try_ffn(ctx, cgraph, i)) { i += 3; pk = PR_FFN; break; }
                if (gen9_try_multi(ctx, cgraph, i)) { pk = PR_QKV; break; }
                {
                    const int used = gen9_try_epi(ctx, cgraph, i);
                    if (used < 0) return GGML_STATUS_FAILED;
                    if (used > 0) { i += used - 1; pk = PR_EPI; break; }
                }
                if (!gen9_mul_mat(ctx, node)) return GGML_STATUS_FAILED;
                pk = PR_MM;
                break;
            case GGML_OP_GLU:
                ew_flush(ctx);
                gen9_glu_alone(node);
                pk = PR_GLU;
                break;
            case GGML_OP_RMS_NORM: {
                if (ctx->ew_cpu) { cpu_node(ctx, cgraph, i); n_ew_cpu_nodes++; pk = PR_EWCPU; break; }
                const int used = gen9_rms_norm(ctx, cgraph, i);
                if (!used) return GGML_STATUS_FAILED;
                i += used - 1; pk = PR_RMS;
                break;
            }
            case GGML_OP_ADD: case GGML_OP_MUL:
                if (ctx->ew_cpu) { cpu_node(ctx, cgraph, i); n_ew_cpu_nodes++; pk = PR_EWCPU; break; }
                if (!gen9_binop(ctx, node)) return GGML_STATUS_FAILED;
                pk = PR_BINOP;
                break;
            case GGML_OP_ROPE:
                if (ctx->ew_cpu) { cpu_node(ctx, cgraph, i); n_ew_cpu_nodes++; pk = PR_EWCPU; break; }
                if (!gen9_rope(ctx, node)) return GGML_STATUS_FAILED;
                pk = PR_ROPE;
                break;
            case GGML_OP_SET_ROWS:
                if (ctx->ew_cpu) { cpu_node(ctx, cgraph, i); n_ew_cpu_nodes++; pk = PR_EWCPU; break; }
                if (!gen9_set_rows(ctx, node)) return GGML_STATUS_FAILED;
                pk = PR_BINOP;   // profiled with ADD/MUL (a copy of a few rows)
                break;
            case GGML_OP_NORM: {
                if (ctx->ew_cpu) { cpu_node(ctx, cgraph, i); n_ew_cpu_nodes++; pk = PR_EWCPU; break; }
                const int used = gen9_layer_norm(ctx, cgraph, i);
                if (!used) return GGML_STATUS_FAILED;
                i += used - 1; pk = PR_NORM;
                break;
            }
            case GGML_OP_UNARY:
                if (ctx->ew_cpu) { cpu_node(ctx, cgraph, i); n_ew_cpu_nodes++; pk = PR_EWCPU; break; }
                if (!gen9_gelu(ctx, node)) return GGML_STATUS_FAILED;
                pk = PR_GELU;
                break;
            case GGML_OP_IM2COL:
                if (ctx->ew_cpu) { cpu_node(ctx, cgraph, i); n_ew_cpu_nodes++; pk = PR_EWCPU; break; }
                if (!gen9_im2col(ctx, node)) return GGML_STATUS_FAILED;
                pk = PR_GELU;
                break;
            case GGML_OP_CPY: case GGML_OP_CONT: case GGML_OP_DUP:
                if (ctx->ew_cpu) { cpu_node(ctx, cgraph, i); n_ew_cpu_nodes++; pk = PR_EWCPU; break; }
                if (!gen9_cpy(ctx, cgraph, i)) return GGML_STATUS_FAILED;
                pk = PR_BINOP;   // profiled with ADD/MUL (memory-bound copies)
                break;
            case GGML_OP_UPSCALE:
                if (ctx->ew_cpu) { cpu_node(ctx, cgraph, i); n_ew_cpu_nodes++; pk = PR_EWCPU; break; }
                if (!gen9_upscale(ctx, node)) return GGML_STATUS_FAILED;
                pk = PR_BINOP;
                break;
            case GGML_OP_FLASH_ATTN_EXT:
                gen9_flash_attn(ctx, cgraph, i);
                pk = PR_FA;
                break;
            default:
                GGML_ABORT("gen9: unsupported op %s reached graph_compute", ggml_op_desc(node));
        }
        if (prof && pk >= 0) {
            if (pk != PR_FA && pk != PR_EWCPU) ew_flush(ctx);   // ew_sync waits already; GEMM paths wait in the library
            const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - pt0).count();
            double fl = 0;
            for (int j = 0; j < cgraph->n_nodes; j++) {   // MUL_MATs computed by this branch (range or marked done)
                const struct ggml_tensor * n = cgraph->nodes[j];
                if (n->op != GGML_OP_MUL_MAT || !((j >= i0 && j <= i) || (!done0.empty() && !done0[j] && ctx->done[j]))) continue;
                fl += 2.0 * (double) n->src[0]->ne[0] * (double) n->ne[0] * (double) n->ne[1] * (double) n->ne[2] * (double) n->ne[3];
            }
            if (pk == PR_FA) fl = 0;
            pr_t[pk] = pr_t[pk] + dt; pr_flop[pk] = pr_flop[pk] + fl; pr_n[pk]++;
        }
    }
    if (ctx->nw) {
        std::lock_guard<std::mutex> lk(gpu_mu);
        if (gen9blas_flush()) { fprintf(stderr, "gen9: flush failed: %s\n", gen9blas_error()); return GGML_STATUS_FAILED; }
        ctx->ew_pending = false;
        const int bad = gen9blas_nowait_pending();
        gen9blas_set_nowait(0); ctx->nw = false;
        if (bad) { n_nw_overflow++; fprintf(stderr, "gen9: %d no-wait products overflowed FP16; graph failed (rerun without GEN9_MM_NOWAIT)\n", bad); return GGML_STATUS_FAILED; }
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

// Smallest batch (tokens, op->ne[1]) taken by MUL_MAT and the fused FFN: GEN9_MIN_BATCH, default 8. Measured crossover
// (results/ggml-backend-min-batch-2026-11-13.txt): with Q4_K_M weights the GPU wins from 5-6 tokens, with F16 from 8
// (tie at 5-7); at 4 tokens and below the CPU wins on all four models tried. Was 32 until 2026-11-13 (design choice).
static int min_batch(void) {
    static const int v = [] { const char * e = getenv("GEN9_MIN_BATCH"); const int b = e ? atoi(e) : 8; return b >= 1 ? b : 8; }();
    return v;
}

// Design section 7: F32 or F16 weights, F32 activations and result.
static bool gen9_supports_mul_mat(const struct ggml_tensor * op) {
    const struct ggml_tensor * w = op->src[0], * x = op->src[1];
    n_mm_asked++;
    const bool w_ok = w->type == GGML_TYPE_F32 || w->type == GGML_TYPE_F16 ||
                      (ggml_is_quantized(w->type) && ggml_get_type_traits(w->type)->to_float != nullptr);   // Q1: dequantized
    const bool x_ok = x->type == GGML_TYPE_F32 || (x->type == GGML_TYPE_F16 && w->type != GGML_TYPE_F32);   // F16: conv_1d (im2col)
    if (!w_ok || !x_ok || op->type != GGML_TYPE_F32) {
        n_mm_declined_type++; return false;
    }
    if (w->ne[2] != 1 || w->ne[3] != 1 || x->ne[2] != 1 || x->ne[3] != 1 || !ggml_is_contiguous(w) || !ggml_is_contiguous(x) ||
        !ggml_is_contiguous(op) || !fits_int(w->ne[0]) || !fits_int(w->ne[1]) || !fits_int(x->ne[1])) { n_mm_declined_shape++; return false; }
    if (op->ne[0] < 32 || op->ne[1] < min_batch() || x->ne[0] < 32) { n_mm_declined_batch++; return false; }
    if (ggml_get_op_params_i32(op, 1) == GGML_HINT_SRC0_IS_HADAMARD) { n_mm_declined_hint++; return false; }
    if (w->type != GGML_TYPE_F32) {   // FP16 chunk sums and FP16 activations: not allowed if F32 / BF16 precision is required
        const enum ggml_prec acc = ggml_prec_get_acc(op), p1 = ggml_prec_get_src(op, 1);
        if (acc == GGML_PREC_F32 || acc == GGML_PREC_BF16 || p1 == GGML_PREC_F32 || p1 == GGML_PREC_BF16) { n_mm_declined_hint++; return false; }
    }
    if (!gpu_ok()) { n_mm_declined_gpu++; return false; }
    return true;
}

// Elementwise ops (see gen9_rms_norm): F32 rows, at least GEN9_EW_MIN_ROWS rows (tokens; default GEN9_MIN_BATCH, as
// MUL_MAT). A 384-row default was tried and dropped: on llama.cpp pp256 the GPU ops win (+1-3.5 %, 3 pairs), so row count
// does not explain whisper's decoder prompt (256 rows, 6 % slower with them; under investigation).
static int ew_min_rows(void) {
    static const int v = [] { const char * e = getenv("GEN9_EW_MIN_ROWS"); return e && atoi(e) >= 1 ? atoi(e) : 0; }();
    return std::max(v, min_batch());
}
static bool gen9_supports_ew(const struct ggml_tensor * op) {
    n_ew_asked++;
    auto f32_rows = [](const struct ggml_tensor * t) {
        return t && t->type == GGML_TYPE_F32 && ggml_is_contiguous(t) && fits_int(t->ne[0]) && fits_int(ggml_nrows(t));
    };
    bool ok = false;
    static const char * skip = getenv("GEN9_EW_SKIP");   // diagnostics: comma list of rms,add,mul,rope left to the CPU
    const char * nm = op->op == GGML_OP_RMS_NORM ? "rms" : op->op == GGML_OP_ADD ? "add" : op->op == GGML_OP_MUL ? "mul" :
                      op->op == GGML_OP_NORM ? "lnorm" : op->op == GGML_OP_UNARY ? "gelu" : op->op == GGML_OP_IM2COL ? "im2col" :
                      op->op == GGML_OP_SET_ROWS ? "setrows" : op->op == GGML_OP_UPSCALE ? "upscale" :
                      (op->op == GGML_OP_CPY || op->op == GGML_OP_CONT || op->op == GGML_OP_DUP) ? "cpy" : "rope";
    if (!ew_off() && !(skip && strstr(skip, nm))) switch (op->op) {
        case GGML_OP_RMS_NORM: {
            float eps; memcpy(&eps, op->op_params, sizeof eps);
            ok = f32_rows(op->src[0]) && f32_rows(op) && ggml_are_same_shape(op->src[0], op) && eps >= 0.0f && ggml_nrows(op) >= ew_min_rows();
            break;
        }
        case GGML_OP_NORM: {
            float eps; memcpy(&eps, op->op_params, sizeof eps);
            ok = f32_rows(op->src[0]) && f32_rows(op) && ggml_are_same_shape(op->src[0], op) && eps >= 0.0f && ggml_nrows(op) >= ew_min_rows();
            break;
        }
        case GGML_OP_IM2COL:
            ok = im2col_ok(op);
            break;
        case GGML_OP_CPY: case GGML_OP_CONT: case GGML_OP_DUP: {   // F32 / F16, any strides, same element count (gen9_cpy)
            const struct ggml_tensor * a = op->src[0];
            auto t_ok = [](const struct ggml_tensor * t) {
                if (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16) return false;
                const size_t es = ggml_type_size(t->type);
                for (int k = 0; k < 4; k++) if (!fits_int(t->ne[k]) || t->nb[k] % es) return false;
                return fits_int(t->ne[0] * t->ne[1]) && fits_int(t->ne[2] * t->ne[3]);
            };
            ok = vision_ops_on() && t_ok(a) && t_ok(op) && ggml_nelements(a) == ggml_nelements(op) && gpu_ok() && gen9blas_has(GEN9BLAS_FEAT_CPY) &&
                 ggml_nrows(op) >= ew_min_rows();
            break;
        }
        case GGML_OP_UPSCALE: {
            const struct ggml_tensor * a = op->src[0];
            const int32_t mf = ggml_get_op_params_i32(op, 0);
            ok = vision_ops_on() && a->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 && ((mf & 0xFF) == 0 || (mf & 0xFF) == 1) &&
                 !(mf & ~0x3FF) && a->nb[0] % 4 == 0 && a->nb[1] % 4 == 0 && a->nb[2] % 4 == 0 && a->nb[3] % 4 == 0 &&
                 op->nb[0] % 4 == 0 && op->nb[1] % 4 == 0 && op->nb[2] % 4 == 0 && op->nb[3] % 4 == 0 &&
                 fits_int(op->ne[0] * op->ne[1]) && fits_int(op->ne[2] * op->ne[3]) && gpu_ok() && gen9blas_has(GEN9BLAS_FEAT_CPY);
            break;
        }
        case GGML_OP_SET_ROWS: {   // F32 rows into an F16 tensor (llama.cpp's KV-cache writes), gen9blas_set_rows_f16
            const struct ggml_tensor * a = op->src[0], * ix = op->src[1];
            static const bool on = [] { const char * e = getenv("GEN9_EW_SETROWS"); return e && !strcmp(e, "1"); }();   // opt-in
            ok = on && a->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F16 && gpu_ok() && gen9blas_has(GEN9BLAS_FEAT_SET_ROWS) &&
                 (ix->type == GGML_TYPE_I64 || ix->type == GGML_TYPE_I32) && a->nb[0] == 4 && op->nb[0] == 2 &&
                 a->nb[1] % 4 == 0 && a->nb[2] % 4 == 0 && a->nb[3] % 4 == 0 && op->nb[1] % 2 == 0 && op->nb[2] % 2 == 0 &&
                 op->nb[3] % 2 == 0 && op->ne[0] == a->ne[0] && op->ne[2] == a->ne[2] && op->ne[3] == a->ne[3] &&
                 ix->ne[0] == a->ne[1] && a->ne[2] % ix->ne[1] == 0 && a->ne[3] % ix->ne[2] == 0 &&
                 fits_int(a->ne[0]) && fits_int(a->ne[1]) && fits_int(a->ne[2] * a->ne[3]);
            break;
        }
        case GGML_OP_UNARY:
            ok = (ggml_get_unary_op(op) == GGML_UNARY_OP_GELU || ggml_get_unary_op(op) == GGML_UNARY_OP_GELU_ERF ||
                  ggml_get_unary_op(op) == GGML_UNARY_OP_RELU) && f32_rows(op->src[0]) && f32_rows(op) &&
                 ggml_are_same_shape(op->src[0], op) && ggml_nrows(op) >= ew_min_rows();
            break;
        case GGML_OP_ADD: case GGML_OP_MUL: {
            const struct ggml_tensor * a = op->src[0], * b = op->src[1];
            ok = f32_rows(a) && f32_rows(b) && f32_rows(op) && ggml_are_same_shape(a, op) && (b->ne[0] == a->ne[0] || b->ne[0] == 1) &&
                 (ggml_nrows(b) == 1 || ggml_are_same_shape(a, b) ||
                  (b->ne[0] == 1 && b->ne[1] == a->ne[1] && b->ne[2] == a->ne[2] && b->ne[3] == a->ne[3])) && ggml_nrows(op) >= ew_min_rows();
            break;
        }
        case GGML_OP_ROPE: {
#ifdef GEN9_GGML_NO_ROPE_TRUNCATE
            break;   // ggml without ggml_rope_get_truncate (whisper.cpp, NeMo-Speech.cpp): ROPE stays on the CPU
                     // (tried on NeMo's Nemotron 3 encoder: GPU ROPE changed the diarization by 10-34 % of frames)
#endif
            const struct ggml_tensor * x = op->src[0], * pos = op->src[1], * ff = op->src[2];
            const int32_t * pp = (const int32_t *) op->op_params;
            const int n_dims = pp[1], mode = pp[2], n_offs = pp[15];
            if (mode == GGML_ROPE_TYPE_VISION) {   // ggml_rope_multi(..., VISION): n_dims = ne0 / 2, all ne0 elements rotated
                int sec[4]; memcpy(sec, pp + 11, sizeof sec);
                ok = vision_ops_on() && x->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 && ggml_is_contiguous(op) &&
                     ggml_are_same_shape(x, op) && 2 * n_dims == x->ne[0] && n_offs == 0 && sec[0] >= 0 && sec[1] >= 0 && sec[2] >= 0 &&
                     sec[3] >= 0 && sec[0] + sec[1] + sec[2] + sec[3] > 0 && x->ne[3] == 1 && x->nb[0] == 4 && x->nb[1] % 4 == 0 && x->nb[2] % 4 == 0 &&
                     x->nb[1] >= (size_t) x->ne[0] * 4 && x->nb[2] >= (x->ne[1] - 1) * x->nb[1] + (size_t) x->ne[0] * 4 &&
                     fits_int(x->ne[1]) && fits_int(x->ne[2]) && n_dims % 2 == 0 &&
                     pos && pos->type == GGML_TYPE_I32 && ggml_is_contiguous(pos) && pos->ne[0] == 4 * x->ne[2] &&
                     (!ff || (ff->type == GGML_TYPE_F32 && ggml_is_contiguous(ff) && ff->ne[0] >= n_dims)) &&
                     gpu_ok() && gen9blas_has(GEN9BLAS_FEAT_ROPE_MULTI) && x->ne[2] >= ew_min_rows();
                break;
            }
            ok = x->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 && ggml_is_contiguous(op) && ggml_are_same_shape(x, op) &&
                 (mode == GGML_ROPE_TYPE_NORMAL || mode == GGML_ROPE_TYPE_NEOX ||
                  (mode == GGML_ROPE_TYPE_MROPE && mrope_on() && gpu_ok() && gen9blas_has(GEN9BLAS_FEAT_ROPE_MULTI))) && n_dims == x->ne[0] && n_offs == 0 &&
                 x->ne[3] == 1 && x->nb[0] == 4 && x->nb[1] % 4 == 0 && x->nb[2] % 4 == 0 &&
                 x->nb[1] >= (size_t) n_dims * 4 && x->nb[2] >= (x->ne[1] - 1) * x->nb[1] + (size_t) n_dims * 4 &&
                 fits_int(x->ne[1]) && fits_int(x->ne[2]) && n_dims % 2 == 0 &&
                 pos && pos->type == GGML_TYPE_I32 && ggml_is_contiguous(pos) &&
                 pos->ne[0] == (mode == GGML_ROPE_TYPE_MROPE ? 4 : 1) * x->ne[2] &&
                 (!ff || (ff->type == GGML_TYPE_F32 && ggml_is_contiguous(ff) && ff->ne[0] >= n_dims / 2)) &&
                 x->ne[2] >= ew_min_rows();
            break;
        }
        default: break;
    }
    if (ok && !gpu_ok()) ok = false;
    if (!ok) n_ew_declined++;
    static const bool dbg = getenv("GEN9_EW_DEBUG") != nullptr;
    if (dbg) fprintf(stderr, "gen9 ew: %s %s %s [%lld %lld %lld %lld] src0 %s %s contig %d\n", ok ? "take" : "decline", ggml_op_desc(op), op->name,
                     (long long) op->ne[0], (long long) op->ne[1], (long long) op->ne[2], (long long) op->ne[3], op->src[0]->name,
                     ggml_type_name(op->src[0]->type), (int) ggml_is_contiguous(op->src[0]));
    return ok;
}

static bool gen9_dev_supports_op(ggml_backend_dev_t, const struct ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_NONE: case GGML_OP_RESHAPE: case GGML_OP_VIEW: case GGML_OP_PERMUTE: case GGML_OP_TRANSPOSE:
            return true;   // no-op nodes: test-backend-ops asks for every tensor of a test graph
        case GGML_OP_MUL_MAT:
            return gen9_supports_mul_mat(op);
        case GGML_OP_FLASH_ATTN_EXT:
            return gen9_supports_fa(op);
        case GGML_OP_RMS_NORM: case GGML_OP_ADD: case GGML_OP_MUL: case GGML_OP_ROPE: case GGML_OP_NORM: case GGML_OP_UNARY:
        case GGML_OP_IM2COL: case GGML_OP_SET_ROWS: case GGML_OP_CPY: case GGML_OP_CONT: case GGML_OP_DUP: case GGML_OP_UPSCALE:
            return gen9_supports_ew(op);
        case GGML_OP_GLU: {   // SWIGLU of two MUL_MAT results: computed inside a fused FFN block (gen9_try_ffn)
            n_glu_asked++;
            const struct ggml_tensor * a = op->src[0], * b = op->src[1];
            const bool ok = ggml_get_glu_op(op) == GGML_GLU_OP_SWIGLU && b && a->type == GGML_TYPE_F32 && b->type == GGML_TYPE_F32 &&
                            op->type == GGML_TYPE_F32 && ggml_is_contiguous(a) && ggml_is_contiguous(b) && ggml_is_contiguous(op) &&
                            ggml_are_same_shape(a, b) && a->ne[2] == 1 && a->ne[3] == 1 && op->ne[1] >= min_batch() &&
                            a->op == GGML_OP_MUL_MAT && b->op == GGML_OP_MUL_MAT && ffn_weight_type(a->src[0]->type) &&
                            ffn_weight_type(b->src[0]->type) && !ffn_off() && gpu_ok();   // only what gen9_try_ffn can fuse
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

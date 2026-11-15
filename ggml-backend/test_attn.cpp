// test_attn: FLASH_ATTN_EXT through the GEN9 backend vs ggml-cpu on the same inputs, in llama.cpp's own layouts
// (docs/attention-gpu.md section 11). test-backend-ops cannot reach the GPU path: its masks are random values
// (not 0 / -inf prefixes) and its K/V are not in the KV-cache row layout, so GEN9 computes them with ggml-cpu.
// Here the graph is built like llm_graph_context::build_attn_mha (llama.cpp 2cbe0ec): K/V views of a KV-cache tensor
// [HKV*D, kv_size] F16, permuted; Q the permuted view of Qcur [D, H, n_q]; mask [n_kv, n_q] F16.
//
// Usage: GEN9_ATTN=1 test_attn <libggml-gen9.so> <D> <H> <HKV> <n_q> <depth> <kv_size> [mode] [qrange]
//   mode: causal (default: token t sees cells 0 .. depth+t), full (no mask), rows (kv_rows = identity list + causal
//         mask over its columns), hole (causal with cell 3 masked for every token: must fall back to ggml-cpu),
//         nonpad (n_kv = depth + n_q, not padded to 256: declined if not a multiple of 16 and masked),
//         whisper (no mask, n_kv = depth + n_q not a multiple of 16, kv_size >= the next multiple: whisper.cpp's
//         encoder, e.g. 64 20 20 1500 0 1536; the rows behind n_kv are zero as in its kv_pad buffer),
//         whispernan (as whisper, V rows n_kv.. set to NaN: the GPU must not be used, the result must stay finite);
//         the whisper shape with mode causal (64 20 20 512 0 1536) must be computed by ggml-cpu ("library declined"):
//         with G 1 < RQ 4 a subgroup holds 4 tokens and the kernel takes the visible key count of the first one;
//         seqs / seqsfull / seqsrows: four sequences in one ubatch (block-diagonal mask, causal or full; seqsrows with
//         an identity kv_rows list); wrap: a ring cache whose sequence starts in the last 12 cells; perm: kv_rows not
//         the identity (reversed 64-row blocks): the general path (fa_mask_runs) gathers K/V rows when needed;
//         an identity kv_rows list), computed as one launch per sequence (fa_mask_blocks in ggml-gen9.cpp);
//         Qwen2.5-7B (128 28 4, G 7, ROWVIS binary): causal at any n_q incl. odd ones; KCL_USERPTR_MB=1 forces the
//         ggml-cpu fallback ("library declined": the K/V region cannot be pinned); depth 16000 / kv_size 16896 splits
//         each node into several launches (GEN9_INFO)
//   qrange: Q uniform in [-qrange, qrange] (default 1; 20 = large logits as test_flash_attn_ext_large_logits)
// Output: supports_op, NMSE (test-backend-ops metric, its FLASH_ATTN_EXT bound is 5e-4), max |gpu - cpu| / max |cpu|,
// wall times. GEN9_INFO=1 prints the backend's counters at exit.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 8) { fprintf(stderr, "usage: %s lib D H HKV n_q depth kv_size [causal|full|rows|hole|nonpad|whisper|whispernan|seqs|seqsfull|seqsrows|wrap|perm] [qrange]\n", argv[0]); return 2; }
    const int D = atoi(argv[2]), H = atoi(argv[3]), HKV = atoi(argv[4]), nq = atoi(argv[5]), depth = atoi(argv[6]), kv_size = atoi(argv[7]);
    const std::string mode = argc > 8 ? argv[8] : "causal";
    const float qrange = argc > 9 ? atof(argv[9]) : 1.0f;
    const bool whisper = mode == "whisper" || mode == "whispernan";
    int n_kv = mode == "nonpad" || whisper ? depth + nq : (depth + nq + 255) / 256 * 256;
    if (n_kv > kv_size) n_kv = kv_size;
    ggml_backend_reg_t reg = ggml_backend_load(argv[1]);
    if (!reg) { fprintf(stderr, "cannot load %s\n", argv[1]); return 1; }
    ggml_backend_t gen9 = ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, 0), nullptr);
    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(cpu, 4);

    ggml_init_params ip = { 64 * ggml_tensor_overhead() + ggml_graph_overhead() * 2, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    const size_t row = (size_t) HKV * D * 2;
    ggml_tensor * kc = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, (int64_t) HKV * D, kv_size);
    ggml_tensor * vc = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, (int64_t) HKV * D, kv_size);
    ggml_tensor * qcur = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, H, nq);
    const bool rows_mode = mode == "rows" || mode == "seqsrows" || mode == "perm";
    ggml_tensor * rows = rows_mode ? ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_kv, 1) : nullptr;
    ggml_tensor * mask = mode == "full" || whisper ? nullptr : ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv, nq);
    // views as in llama_kv_cache::get_k / get_v (flash attention: V not transposed) and build_attn_mha
    const int64_t nrows = rows_mode ? kv_size : n_kv;
    ggml_tensor * k = ggml_view_4d(ctx, kc, D, HKV, nrows, 1, D * 2, row, row * nrows, 0);
    ggml_tensor * v = ggml_view_4d(ctx, vc, D, HKV, nrows, 1, D * 2, row, row * nrows, 0);
    ggml_tensor * q = ggml_permute(ctx, qcur, 0, 2, 1, 3);
    k = ggml_permute(ctx, k, 0, 2, 1, 3);
    v = ggml_permute(ctx, v, 0, 2, 1, 3);
    const float scale = 1.0f / sqrtf((float) D);
    ggml_tensor * og, * oc;
    if (rows_mode) {
        og = ggml_flash_attn_ext_rows(ctx, q, k, v, mask, rows, scale, 0.0f, 0.0f);
        oc = ggml_flash_attn_ext_rows(ctx, q, k, v, mask, rows, scale, 0.0f, 0.0f);
    } else {
        og = ggml_flash_attn_ext(ctx, q, k, v, mask, scale, 0.0f, 0.0f);
        oc = ggml_flash_attn_ext(ctx, q, k, v, mask, scale, 0.0f, 0.0f);
    }
    ggml_prec_set_acc(og, GGML_PREC_F32); ggml_prec_set_acc(oc, GGML_PREC_F32);   // as llama.cpp
    ggml_cgraph * gg = ggml_new_graph(ctx); ggml_build_forward_expand(gg, og);
    ggml_cgraph * gc = ggml_new_graph(ctx); ggml_build_forward_expand(gc, oc);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, cpu);
    ggml_backend_buffer_clear(buf, 0);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<ggml_fp16_t> h((size_t) HKV * D * kv_size);
    for (auto & x : h) { x = ggml_fp32_to_fp16(u(rng)); } memcpy(kc->data, h.data(), ggml_nbytes(kc));
    for (auto & x : h) { x = ggml_fp32_to_fp16(u(rng)); } memcpy(vc->data, h.data(), ggml_nbytes(vc));
    if (whisper) {   // rows behind n_kv: zero (whisper.cpp clears kv_pad once), or NaN in V for whispernan
        memset((char *) kc->data + (size_t) n_kv * row, 0, (size_t) (kv_size - n_kv) * row);
        ggml_fp16_t f = ggml_fp32_to_fp16(mode == "whispernan" ? NAN : 0.0f);
        for (size_t i = (size_t) n_kv * HKV * D; i < (size_t) kv_size * HKV * D; i++) ((ggml_fp16_t *) vc->data)[i] = f;
    }
    for (int64_t i = 0; i < ggml_nelements(qcur); i++) ((float *) qcur->data)[i] = qrange * u(rng);
    if (rows) for (int c = 0; c < n_kv; c++) ((int32_t *) rows->data)[c] = c < depth + nq ? c : -1;
    if (mode == "perm")   // kv_rows not the identity: columns read the cache rows in reversed 64-row blocks
        for (int c = 0; c < depth + nq && c < n_kv; c++) { int b = c / 64, o = c % 64; int r = b * 64 + 63 - o; ((int32_t *) rows->data)[c] = r < depth + nq ? r : c; }
    if (mask) {
        ggml_fp16_t z = ggml_fp32_to_fp16(0.0f), ninf = ggml_fp32_to_fp16(-INFINITY);
        // seqs*: 4 sequences of unequal length in one ubatch (llama-server embeddings), cells contiguous per
        // sequence from depth; token at position p of sequence s sees cells [depth + start_s, depth + start_s + p]
        // (seqsfull: the whole sequence, no causal limit)
        const bool seqs = mode.rfind("seqs", 0) == 0;
        int st[5] = { 0, nq * 2 / 5, nq * 13 / 20, nq * 17 / 20, nq };
        for (int t = 0; t < nq; t++)
            for (int c = 0; c < n_kv; c++) {
                bool vis = c <= depth + t;
                if (mode == "wrap")   // ring cache: the sequence's first 12 tokens sit in the last 12 cells, the rest from cell 0
                    vis = (c >= n_kv - 12) || c <= t;
                if (seqs) {
                    int sq = 0; while (t >= st[sq + 1]) sq++;
                    const int lo = depth + st[sq], hi = mode == "seqsfull" ? depth + st[sq + 1] : depth + t + 1;
                    vis = c >= lo && c < hi;
                }
                if (mode == "hole" && c == 3) vis = false;
                ((ggml_fp16_t *) mask->data)[(size_t) t * n_kv + c] = vis ? z : ninf;
            }
    }
    printf("D=%d H=%d HKV=%d n_q=%d depth=%d kv_size=%d n_kv=%d mode=%s qrange=%g: GEN9 supports_op %d\n", D, H, HKV, nq, depth,
           kv_size, n_kv, mode.c_str(), qrange, (int) ggml_backend_supports_op(gen9, og));
    auto now = [] { return std::chrono::steady_clock::now(); };
    auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    if (!ggml_backend_supports_op(gen9, og)) {   // TEST_ATTN_CPU=1: time ggml-cpu alone (shapes without a GPU kernel yet)
        if (!getenv("TEST_ATTN_CPU")) return 0;
        double tc = 1e30;
        for (int r = 0; r < 3; r++) { auto b = now(); ggml_backend_graph_compute(cpu, gc); tc = std::min(tc, ms(b, now())); }
        printf("  best of 3: ggml-cpu %.2f ms\n", tc);
        return 0;
    }
    double tg = 1e30, tc = 1e30;
    for (int r = 0; r < 3; r++) {
        memset(og->data, 0xff, ggml_nbytes(og));   // NaN: anything not written shows up
        auto a = now(); if (ggml_backend_graph_compute(gen9, gg) != GGML_STATUS_SUCCESS) { fprintf(stderr, "GEN9 compute failed\n"); return 1; }
        auto b = now(); ggml_backend_graph_compute(cpu, gc); auto c = now();
        tg = std::min(tg, ms(a, b)); tc = std::min(tc, ms(b, c));
    }
    const float * g = (const float *) og->data, * c = (const float *) oc->data;
    double se = 0, sr = 0, maxe = 0, maxr = 0; long bad = 0;
    for (int64_t i = 0; i < ggml_nelements(og); i++) {
        double e = (double) g[i] - c[i];
        if (!std::isfinite(g[i])) bad++;
        se += e * e; sr += (double) c[i] * c[i];
        maxe = std::max(maxe, fabs(e)); maxr = std::max(maxr, (double) fabs(c[i]));
    }
    const double nmse = se / sr;
    printf("  NMSE %.3e (bound 5e-4) max|err|/max|ref| %.3e non-finite %ld -> %s | best of 3: GEN9 %.2f ms, ggml-cpu %.2f ms\n",
           nmse, maxe / maxr, bad, (nmse <= 5e-4 && !bad) ? "OK" : "FAIL", tg, tc);
    ggml_backend_buffer_free(buf); ggml_free(ctx); ggml_backend_free(gen9); ggml_backend_free(cpu);
    return nmse <= 5e-4 && !bad ? 0 : 1;
}

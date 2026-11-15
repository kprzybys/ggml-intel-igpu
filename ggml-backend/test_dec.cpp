// test_dec: FLASH_ATTN_EXT over KV streams (llama-server -np N decoding) through the GEN9 backend vs ggml-cpu, in
// llama.cpp's layouts (2cbe0ec, llama_kv_cache with n_stream > 1): cache tensor [HKV*D, kv_size, ns] F16, K/V views
// [D, HKV, n_kv, ns] permuted to [D, n_kv, HKV, ns]; Q the permuted view of Qcur [D, H, n_q, ns]; mask [n_kv, n_q padded
// to GGML_KQ_MASK_PAD, 1, ns] F16 with a different visible prefix per stream (len_s keys, + t for token t: causal).
//
// Usage: test_dec <libggml-gen9.so> <D> <H> <HKV> <ns> <n_q> <max_len> <kv_size> [mode]
//   stream s sees len_s = max_len * (s + 1) / ns keys before its first token (different lengths per stream);
//   mode: causal (default), hole (one stream has a masked cell inside its prefix: must fall back to ggml-cpu)
// Output: supports_op, NMSE (bound 5e-4), max |gpu - cpu| / max |cpu|, best-of-5 times. GEN9_INFO=1: backend counters.
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
    if (argc < 9) { fprintf(stderr, "usage: %s lib D H HKV ns n_q max_len kv_size [causal|hole]\n", argv[0]); return 2; }
    const int D = atoi(argv[2]), H = atoi(argv[3]), HKV = atoi(argv[4]), ns = atoi(argv[5]), nq = atoi(argv[6]);
    const int max_len = atoi(argv[7]), kv_size = atoi(argv[8]);
    const std::string mode = argc > 9 ? argv[9] : "causal";
    const int n_kv = std::min(kv_size, (max_len + nq + 255) / 256 * 256), nqp = (nq + 63) / 64 * 64;   // GGML_KQ_MASK_PAD 64
    ggml_backend_reg_t reg = ggml_backend_load(argv[1]);
    if (!reg) { fprintf(stderr, "cannot load %s\n", argv[1]); return 1; }
    ggml_backend_t gen9 = ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, 0), nullptr);
    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(cpu, 4);
    ggml_init_params ip = { 64 * ggml_tensor_overhead() + ggml_graph_overhead() * 2, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    const size_t row = (size_t) HKV * D * 2;
    ggml_tensor * kc = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, (int64_t) HKV * D, kv_size, ns);
    ggml_tensor * vc = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, (int64_t) HKV * D, kv_size, ns);
    ggml_tensor * qcur = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, H, nq, ns);
    ggml_tensor * mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_kv, nqp, 1, ns);
    ggml_tensor * k = ggml_view_4d(ctx, kc, D, HKV, n_kv, ns, D * 2, row, row * kv_size, 0);
    ggml_tensor * v = ggml_view_4d(ctx, vc, D, HKV, n_kv, ns, D * 2, row, row * kv_size, 0);
    ggml_tensor * q = ggml_permute(ctx, qcur, 0, 2, 1, 3);
    k = ggml_permute(ctx, k, 0, 2, 1, 3);
    v = ggml_permute(ctx, v, 0, 2, 1, 3);
    const float scale = 1.0f / sqrtf((float) D);
    ggml_tensor * og = ggml_flash_attn_ext(ctx, q, k, v, mask, scale, 0.0f, 0.0f);
    ggml_tensor * oc = ggml_flash_attn_ext(ctx, q, k, v, mask, scale, 0.0f, 0.0f);
    ggml_prec_set_acc(og, GGML_PREC_F32); ggml_prec_set_acc(oc, GGML_PREC_F32);
    ggml_cgraph * gg = ggml_new_graph(ctx); ggml_build_forward_expand(gg, og);
    ggml_cgraph * gc = ggml_new_graph(ctx); ggml_build_forward_expand(gc, oc);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, cpu);
    ggml_backend_buffer_clear(buf, 0);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    for (int64_t i = 0; i < ggml_nelements(kc); i++) ((ggml_fp16_t *) kc->data)[i] = ggml_fp32_to_fp16(u(rng));
    for (int64_t i = 0; i < ggml_nelements(vc); i++) ((ggml_fp16_t *) vc->data)[i] = ggml_fp32_to_fp16(u(rng));
    for (int64_t i = 0; i < ggml_nelements(qcur); i++) ((float *) qcur->data)[i] = 2.0f * u(rng);
    const ggml_fp16_t z = ggml_fp32_to_fp16(0.0f), ninf = ggml_fp32_to_fp16(-INFINITY);
    long keys = 0;
    for (int s = 0; s < ns; s++) {
        const int len = std::max(1, max_len * (s + 1) / ns);
        for (int t = 0; t < nqp; t++)
            for (int c = 0; c < n_kv; c++) {
                bool vis = t < nq && c < len + t;
                if (mode == "hole" && s == ns / 2 && c == 3) vis = false;
                ((ggml_fp16_t *) mask->data)[((size_t) s * nqp + t) * n_kv + c] = vis ? z : ninf;
                if (vis) keys++;
            }
    }
    printf("D=%d H=%d HKV=%d ns=%d n_q=%d max_len=%d kv_size=%d n_kv=%d mode=%s: GEN9 supports_op %d\n", D, H, HKV, ns, nq, max_len,
           kv_size, n_kv, mode.c_str(), (int) ggml_backend_supports_op(gen9, og));
    if (!ggml_backend_supports_op(gen9, og)) return 1;
    auto now = [] { return std::chrono::steady_clock::now(); };
    auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    double tg = 1e30, tc = 1e30;
    for (int r = 0; r < 5; r++) {
        memset(og->data, 0xff, ggml_nbytes(og));
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
    const double nmse = se / sr, gflop = 4.0 * D * (double) keys * (H / HKV) * HKV / 1e9;
    printf("  NMSE %.3e (bound 5e-4) max|err|/max|ref| %.3e non-finite %ld -> %s | best of 5: GEN9 %.3f ms, ggml-cpu %.3f ms (%.2f GFLOP, K/V %.1f MB)\n",
           nmse, maxe / maxr, bad, (nmse <= 5e-4 && !bad) ? "OK" : "FAIL", tg, tc, gflop, 2.0 * row * (double) keys / nq / 1e6);
    ggml_backend_buffer_free(buf); ggml_free(ctx); ggml_backend_free(gen9); ggml_backend_free(cpu);
    return nmse <= 5e-4 && !bad ? 0 : 1;
}

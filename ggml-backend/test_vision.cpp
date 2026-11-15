// test_vision: the PaddleOCR-VL vision-encoder ops through the GEN9 backend vs ggml-cpu, in the exact shapes and
// parameters of mtmd (llama.cpp 2cbe0ec, tools/mtmd/models/paddleocr.cpp, mtmd-graph.cpp): an A4 page at 1.0 MP is
// 60 x 84 patches = 5040 tokens, 16 heads x 72.
//
// Usage: test_vision <libggml-gen9.so> <case> [w h]      (w x h patches, default 60 84)
//   rope     ggml_rope_multi(Kcur [72, 16, n], positions [4n], n_dims 36, sections {18,18,18,18}, VISION, base 10000)
//   cast     ggml_cast(ggml_permute(Kcur, 0, 2, 1, 3), F16)              (the K / V copies before flash attention)
//   cont     ggml_cont of a permuted F32 tensor (patch embedding / merge)
//   upscale  27 x 27 learned position embeddings [1152, 27, 27] -> [1152, w, h]: permute, ggml_interpolate bilinear
//            + antialias, permute, ggml_cont_2d (resize_position_embeddings)
//   block    ROPE on Q and K, permutes, K / V cast to F16, flash attention without mask (one encoder layer's attention)
// Output: supports_op of every node, NMSE, max |gpu - cpu| / max |cpu|, bit-identical element share, best-of-3 times.
// GEN9_INFO=1 prints the backend's counters at exit.
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

static ggml_tensor * build(ggml_context * ctx, const std::string & c, int w, int h, std::vector<ggml_tensor *> & ins) {
    const int D = 72, H = 16, n = w * h;
    auto rope = [&](ggml_tensor * x, ggml_tensor * pos) {
        int sec[4] = { D / 4, D / 4, D / 4, D / 4 };
        return ggml_rope_multi(ctx, x, pos, nullptr, D / 2, sec, GGML_ROPE_TYPE_VISION, 32768, 10000, 1, 0, 1, 32, 1);
    };
    if (c == "rope" || c == "cast" || c == "block") {
        ggml_tensor * k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, H, n); ins.push_back(k);
        if (c == "cast") return ggml_cast(ctx, ggml_permute(ctx, k, 0, 2, 1, 3), GGML_TYPE_F16);
        ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4 * n); ins.push_back(pos);
        if (c == "rope") return rope(k, pos);
        ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, H, n); ins.push_back(q);
        ggml_tensor * v = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, H, n); ins.push_back(v);
        ggml_tensor * qp = ggml_permute(ctx, rope(q, pos), 0, 2, 1, 3);
        ggml_tensor * kp = ggml_cast(ctx, ggml_permute(ctx, rope(k, pos), 0, 2, 1, 3), GGML_TYPE_F16);
        ggml_tensor * vp = ggml_cast(ctx, ggml_permute(ctx, v, 0, 2, 1, 3), GGML_TYPE_F16);
        ggml_tensor * o = ggml_flash_attn_ext(ctx, qp, kp, vp, nullptr, 1.0f / sqrtf((float) D), 0.0f, 0.0f);
        ggml_prec_set_acc(o, GGML_PREC_F32);
        return o;
    }
    if (c == "cont") {
        ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1152 * 2, w / 2, h); ins.push_back(x);
        return ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));
    }
    if (c == "upscale") {
        const int E = 1152, s = 27;
        ggml_tensor * pe = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, E, s * s); ins.push_back(pe);
        ggml_tensor * x = ggml_reshape_3d(ctx, pe, E, s, s);
        x = ggml_permute(ctx, x, 2, 0, 1, 3);
        x = ggml_interpolate(ctx, x, w, h, E, 1, GGML_SCALE_MODE_BILINEAR | GGML_SCALE_FLAG_ANTIALIAS);
        x = ggml_permute(ctx, x, 1, 2, 0, 3);
        return ggml_cont_2d(ctx, x, E, w * h);
    }
    return nullptr;
}

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s lib rope|cast|cont|upscale|block [w h]\n", argv[0]); return 2; }
    const std::string cs = argv[2];
    const int w = argc > 4 ? atoi(argv[3]) : 60, h = argc > 4 ? atoi(argv[4]) : 84, n = w * h;
    ggml_backend_reg_t reg = ggml_backend_load(argv[1]);
    if (!reg) { fprintf(stderr, "cannot load %s\n", argv[1]); return 1; }
    ggml_backend_t gen9 = ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, 0), nullptr);
    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(cpu, 4);
    ggml_init_params ip = { 128 * ggml_tensor_overhead() + ggml_graph_overhead() * 2, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    std::vector<ggml_tensor *> ig, ic;
    ggml_tensor * og = build(ctx, cs, w, h, ig), * oc = build(ctx, cs, w, h, ic);
    if (!og) { fprintf(stderr, "unknown case %s\n", cs.c_str()); return 2; }
    ggml_cgraph * gg = ggml_new_graph(ctx); ggml_build_forward_expand(gg, og);
    ggml_cgraph * gc = ggml_new_graph(ctx); ggml_build_forward_expand(gc, oc);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, cpu);
    ggml_backend_buffer_clear(buf, 0);
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    for (size_t j = 0; j < ig.size(); j++) {   // the same random inputs for both graphs
        ggml_tensor * a = ig[j], * b = ic[j];
        if (a->type == GGML_TYPE_I32) {   // mtmd's positions for a w x h grid: (y, x, y, x) per patch
            int32_t * p = (int32_t *) a->data;
            for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) {
                const int i = y * w + x; p[i] = y; p[i + n] = x; p[i + 2 * n] = y; p[i + 3 * n] = x;
            }
        } else for (int64_t e = 0; e < ggml_nelements(a); e++) ((float *) a->data)[e] = (cs == "block" ? 3.0f : 1.0f) * u(rng);
        memcpy(b->data, a->data, ggml_nbytes(a));
    }
    int sup = 1;
    for (int i = 0; i < ggml_graph_n_nodes(gg); i++) {
        ggml_tensor * t = ggml_graph_node(gg, i);
        const int s = ggml_backend_supports_op(gen9, t);
        printf("  node %-12s %-24s GEN9 supports_op %d\n", ggml_op_desc(t), t->name, s);
        sup &= s;
    }
    if (!sup) { printf("%s: not all nodes supported\n", cs.c_str()); return 1; }
    auto now = [] { return std::chrono::steady_clock::now(); };
    auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    double tg = 1e30, tc = 1e30;
    for (int r = 0; r < 3; r++) {
        memset(og->data, 0xff, ggml_nbytes(og));
        auto a = now(); if (ggml_backend_graph_compute(gen9, gg) != GGML_STATUS_SUCCESS) { fprintf(stderr, "GEN9 compute failed\n"); return 1; }
        auto b = now(); ggml_backend_graph_compute(cpu, gc); auto c = now();
        tg = std::min(tg, ms(a, b)); tc = std::min(tc, ms(b, c));
    }
    auto val = [](const ggml_tensor * t, int64_t i) {
        return t->type == GGML_TYPE_F16 ? ggml_fp16_to_fp32(((const ggml_fp16_t *) t->data)[i]) : ((const float *) t->data)[i];
    };
    auto bits = [](const ggml_tensor * t, int64_t i) {
        return t->type == GGML_TYPE_F16 ? (uint32_t) ((const uint16_t *) t->data)[i] : ((const uint32_t *) t->data)[i];
    };
    double se = 0, sr = 0, maxe = 0, maxr = 0; long bad = 0, same = 0;
    const int64_t N = ggml_nelements(og);
    for (int64_t i = 0; i < N; i++) {
        const double g = val(og, i), c = val(oc, i), e = g - c;
        if (!std::isfinite(g)) bad++;
        same += bits(og, i) == bits(oc, i);
        se += e * e; sr += c * c; maxe = std::max(maxe, fabs(e)); maxr = std::max(maxr, fabs(c));
    }
    const double nmse = se / sr, bound = cs == "block" ? 5e-4 : 1e-7;
    printf("%s (%d x %d patches): NMSE %.3e (bound %.0e) max|err|/max|ref| %.3e bit-identical %.4f non-finite %ld -> %s | best of 3: GEN9 %.2f ms, ggml-cpu %.2f ms\n",
           cs.c_str(), w, h, nmse, bound, maxe / maxr, (double) same / N, bad, (nmse <= bound && !bad) ? "OK" : "FAIL", tg, tc);
    ggml_backend_buffer_free(buf); ggml_free(ctx); ggml_backend_free(gen9); ggml_backend_free(cpu);
    return nmse <= bound && !bad ? 0 : 1;
}

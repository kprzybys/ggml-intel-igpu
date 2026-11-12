// Step 1b (docs/ggml-backend-design.md, section 4): can MUL_MAT run in FP16 with gen9blas HS (FP16 operands,
// GEN9BLAS_KC = 32 products summed in FP16, then added into FP32) without overflow? CPU only.
// For every MUL_MAT of a model over a text, per weight: max |src1| (activations), max |w|, the bound
// max|x| * max|w| * 32, and the largest |sum of 32 consecutive products| actually seen on sampled rows (operands
// rounded to FP16, sum in FP32, so the magnitude an FP16 accumulator would have to hold). FP16 max is 65504.
// Usage: f16_range <model.gguf> <text file> [chunks = 10] [ctx = 512]
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

struct stat_t { double max_x = 0, max_w = -1, max_chunk = 0; long calls = 0; int64_t K = 0, N = 0; std::string type; };
static std::map<std::string, stat_t> stats;
static const int KC = 32;

static float f16r(float v) { return ggml_fp16_to_fp32(ggml_fp32_to_fp16(v)); }
static float wget(const ggml_tensor * w, int64_t n, int64_t k) {
    const char * row = (const char *) w->data + n * w->nb[1];
    return w->type == GGML_TYPE_F16 ? ggml_fp16_to_fp32(((const ggml_fp16_t *) row)[k]) : ((const float *) row)[k];
}

static bool cb(ggml_tensor * t, bool ask, void *) {
    if (ask) return t->op == GGML_OP_MUL_MAT;
    if (t->op != GGML_OP_MUL_MAT) return true;
    const ggml_tensor * w = t->src[0], * x = t->src[1];
    if (!w->buffer || !x->buffer || !ggml_backend_buffer_is_host(w->buffer) || !ggml_backend_buffer_is_host(x->buffer) ||
        (w->type != GGML_TYPE_F16 && w->type != GGML_TYPE_F32) || x->type != GGML_TYPE_F32 || w->ne[2] != 1 || x->ne[2] != 1)
        return true;   // only the 2-D weight MUL_MATs the backend would take
    const int64_t K = w->ne[0], N = w->ne[1], M = x->ne[1];
    stat_t & s = stats[w->name];
    s.calls++; s.K = K; s.N = N; s.type = ggml_type_name(w->type);
    if (s.max_w < 0) { double m = 0; for (int64_t n = 0; n < N; n++) for (int64_t k = 0; k < K; k++) m = std::max(m, (double) fabsf(wget(w, n, k))); s.max_w = m; }
    for (int64_t i = 0; i < M; i++) {
        const float * xr = (const float *) ((const char *) x->data + i * x->nb[1]);
        for (int64_t k = 0; k < K; k++) s.max_x = std::max(s.max_x, (double) fabsf(xr[k]));
    }
    const int64_t rows[4] = { 0, M / 3, 2 * M / 3, M - 1 };   // sampled activation rows: all N outputs, all chunks
    for (int64_t r : rows) {
        const float * xr = (const float *) ((const char *) x->data + r * x->nb[1]);
        for (int64_t n = 0; n < N; n++)
            for (int64_t k0 = 0; k0 < K; k0 += KC) {
                float acc = 0;
                for (int64_t k = k0; k < std::min(K, k0 + KC); k++) acc += f16r(xr[k]) * f16r(wget(w, n, k));
                s.max_chunk = std::max(s.max_chunk, (double) fabsf(acc));
            }
    }
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s model.gguf text.txt [chunks] [ctx]\n", argv[0]); return 2; }
    int chunks = argc > 3 ? atoi(argv[3]) : 10, n_ctx = argc > 4 ? atoi(argv[4]) : 512;
    llama_backend_init();
    llama_model * model = llama_model_load_from_file(argv[1], llama_model_default_params());
    if (!model) return 1;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx; cp.n_batch = n_ctx; cp.n_ubatch = n_ctx; cp.cb_eval = cb; cp.cb_eval_user_data = nullptr;
    llama_context * ctx = llama_init_from_model(model, cp);
    FILE * f = fopen(argv[2], "rb"); if (!f) return 1;
    std::string text; char buf[65536]; size_t r; while ((r = fread(buf, 1, sizeof buf, f)) > 0) text.append(buf, r); fclose(f);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> tok(text.size() + 16);
    int nt = llama_tokenize(vocab, text.c_str(), (int) text.size(), tok.data(), (int) tok.size(), true, false);
    if (nt < 0) return 1;
    tok.resize(nt);
    chunks = std::min(chunks, nt / n_ctx);
    for (int c = 0; c < chunks; c++) {
        llama_memory_clear(llama_get_memory(ctx), true);
        if (llama_decode(ctx, llama_batch_get_one(tok.data() + (size_t) c * n_ctx, n_ctx))) { fprintf(stderr, "decode failed\n"); return 1; }
        fprintf(stderr, "chunk %d/%d done\n", c + 1, chunks);
    }
    printf("# f16_range: %s, %d chunks x %d tokens; FP16 max 65504; HS sums %d products in FP16\n", argv[1], chunks, n_ctx, KC);
    printf("%-28s %4s %6s %6s %5s %10s %8s %12s %12s\n", "weight", "type", "N", "K", "calls", "max|x|", "max|w|", "bound x*w*32", "max chunk");
    double gx = 0, gb = 0, gc = 0;
    for (auto & [name, s] : stats) {
        double bound = s.max_x * s.max_w * KC;
        printf("%-28s %4s %6lld %6lld %5ld %10.3g %8.3g %12.3g %12.3g\n", name.c_str(), s.type.c_str(), (long long) s.N,
               (long long) s.K, s.calls, s.max_x, s.max_w, bound, s.max_chunk);
        gx = std::max(gx, s.max_x); gb = std::max(gb, bound); gc = std::max(gc, s.max_chunk);
    }
    printf("# overall: max|x| %.4g, max bound %.4g, max chunk seen %.4g (FP16 limit 65504)\n", gx, gb, gc);
    llama_free(ctx); llama_model_free(model);
    return 0;
}

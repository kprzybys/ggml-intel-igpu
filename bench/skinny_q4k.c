// Test and timing of bench/skinny_q4k.cl (results/skinny-q4k-model-2026-11-13.txt): random Q4_K weights (ggml block
// layout), random FP16 activations, FP64 reference from ggml's dequantize_row_q4_K formula.
// Usage: skinny_q4k <N> <K> <KT> <RW> [reps=20] [version 1|2; 2 = skinny_q4k_t, lanes = tokens, F32 activations]   -> best kernel time, weight GB/s, max error / sum |w a|.
// Build: gcc -O2 -Wall -o skinny_q4k skinny_q4k.c -lOpenCL (needs a compiler with _Float16).
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { cl_int e_ = (x); if (e_ != CL_SUCCESS) { fprintf(stderr, "%s:%d: %s = %d\n", __FILE__, __LINE__, #x, e_); exit(1); } } while (0)

static uint64_t rs = 88172645463325252ull;
static uint32_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (uint32_t)rs; }
static float frnd(float lo, float hi) { return lo + (hi - lo) * (rnd() / 4294967296.0f); }

static void scale_min(int j, const uint8_t *q, int *sc, int *m) {   // ggml get_scale_min_k4
    if (j < 4) { *sc = q[j] & 63; *m = q[j + 4] & 63; }
    else { *sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}
static void dequant_block(const uint8_t *blk, double *y) {          // ggml dequantize_row_q4_K, one block
    _Float16 hd, hm; memcpy(&hd, blk, 2); memcpy(&hm, blk + 2, 2);
    const double d = (double)hd, dmin = (double)hm;
    const uint8_t *sc = blk + 4, *q = blk + 16;
    int is = 0;
    for (int j = 0; j < 256; j += 64) {
        int s1, m1, s2, m2; scale_min(is, sc, &s1, &m1); scale_min(is + 1, sc, &s2, &m2);
        for (int l = 0; l < 32; l++) y[j + l] = d * s1 * (q[l] & 0xF) - dmin * m1;
        for (int l = 0; l < 32; l++) y[j + 32 + l] = d * s2 * (q[l] >> 4) - dmin * m2;
        q += 32; is += 2;
    }
}

int main(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s N K KT RW [reps]\n", argv[0]); return 2; }
    const int N = atoi(argv[1]), K = atoi(argv[2]), KT = atoi(argv[3]), RW = atoi(argv[4]), reps = argc > 5 ? atoi(argv[5]) : 20;
    const int ver = argc > 6 ? atoi(argv[6]) : 1, SPLIT = argc > 7 ? atoi(argv[7]) : 1;
    if (K % 256 || N % RW) { fprintf(stderr, "K %% 256 and N %% RW must be 0\n"); return 2; }
    const size_t rb = (size_t)K / 256 * 144, wbytes = rb * N;
    uint8_t *W = malloc(wbytes);
    for (size_t n = 0; n < (size_t)N; n++)
        for (int b = 0; b < K / 256; b++) {
            uint8_t *blk = W + n * rb + (size_t)b * 144;
            _Float16 d = (_Float16)frnd(1e-3f, 4e-3f), m = (_Float16)frnd(1e-3f, 4e-3f);
            memcpy(blk, &d, 2); memcpy(blk + 2, &m, 2);
            for (int i = 4; i < 144; i++) blk[i] = (uint8_t)rnd();
        }
    _Float16 *A = malloc(sizeof(_Float16) * KT * K);
    float *Af = malloc(sizeof(float) * KT * K);
    for (int i = 0; i < KT * K; i++) { A[i] = (_Float16)frnd(-1.f, 1.f); Af[i] = (float)A[i]; }

    cl_platform_id plat; cl_device_id dev; cl_int err;
    CHECK(clGetPlatformIDs(1, &plat, NULL)); CHECK(clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, NULL));
    cl_context ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &err); CHECK(err);
    cl_queue_properties qp[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    cl_command_queue q = clCreateCommandQueueWithProperties(ctx, dev, qp, &err); CHECK(err);
    FILE *f = fopen("skinny_q4k.cl", "rb"); if (!f) { perror("skinny_q4k.cl"); return 1; }
    fseek(f, 0, SEEK_END); long fl = ftell(f); rewind(f); char *src = malloc(fl + 1);
    if (fread(src, 1, fl, f) != (size_t)fl) return 1;
    src[fl] = 0; fclose(f);
    cl_program prog = clCreateProgramWithSource(ctx, 1, (const char **)&src, NULL, &err); CHECK(err);
    char opts[128]; snprintf(opts, sizeof opts, "-DKT=%d -DRW=%d -DSPLIT=%d", KT, RW, SPLIT);
    if (clBuildProgram(prog, 1, &dev, opts, NULL, NULL) != CL_SUCCESS) {
        static char log[1 << 16]; clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, sizeof log, log, NULL);
        fprintf(stderr, "build error:\n%s\n", log); return 1;
    }
    cl_kernel k = clCreateKernel(prog, ver == 2 ? "skinny_q4k_t" : "skinny_q4k", &err); CHECK(err);
    cl_mem bW = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, wbytes, W, &err); CHECK(err);
    cl_mem bA = ver == 2 ? clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * KT * K, Af, &err)
                         : clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(_Float16) * KT * K, A, &err); CHECK(err);
    cl_mem bY = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sizeof(float) * KT * N * SPLIT, NULL, &err); CHECK(err);
    cl_ulong rbl = rb;
    CHECK(clSetKernelArg(k, 0, sizeof bW, &bW)); CHECK(clSetKernelArg(k, 1, sizeof rbl, &rbl));
    CHECK(clSetKernelArg(k, 2, sizeof bA, &bA)); CHECK(clSetKernelArg(k, 3, sizeof bY, &bY));
    CHECK(clSetKernelArg(k, 4, sizeof N, &N)); CHECK(clSetKernelArg(k, 5, sizeof K, &K));
    size_t gl[2] = { (size_t)(N / RW) * 8, ver == 2 ? (size_t)SPLIT : 1 }, lo[2] = { 64, 1 };
    while (gl[0] % lo[0]) lo[0] /= 2;
    double best = 1e30;
    for (int r = 0; r <= reps; r++) {
        cl_event ev;
        CHECK(clEnqueueNDRangeKernel(q, k, 2, NULL, gl, lo, 0, NULL, &ev)); CHECK(clWaitForEvents(1, &ev));
        cl_ulong t0, t1;
        CHECK(clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof t0, &t0, NULL));
        CHECK(clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof t1, &t1, NULL));
        if (r > 0 && (t1 - t0) * 1e-6 < best) best = (t1 - t0) * 1e-6;
        clReleaseEvent(ev);
    }
    const int parts = ver == 2 ? SPLIT : 1;
    float *Y = calloc((size_t)KT * N, sizeof(float)), *Yp = malloc(sizeof(float) * KT * N * parts);
    CHECK(clEnqueueReadBuffer(q, bY, CL_TRUE, 0, sizeof(float) * KT * N * parts, Yp, 0, NULL, NULL));
    for (int s = 0; s < parts; s++) for (size_t i = 0; i < (size_t)KT * N; i++) Y[i] += Yp[(size_t)s * KT * N + i];   // partial sums
    double *w = malloc(sizeof(double) * K), maxerr = 0;
    for (int n = 0; n < N; n++) {
        for (int b = 0; b < K / 256; b++) dequant_block(W + (size_t)n * rb + (size_t)b * 144, w + 256 * b);
        for (int t = 0; t < KT; t++) {
            double s = 0, sa = 0;
            for (int i = 0; i < K; i++) { s += w[i] * (double)A[t * K + i]; sa += fabs(w[i] * (double)A[t * K + i]); }
            const double e = fabs(Y[(size_t)t * N + n] - s) / (sa > 0 ? sa : 1);
            if (e > maxerr || isnan(Y[(size_t)t * N + n])) maxerr = isnan(Y[(size_t)t * N + n]) ? INFINITY : e;
        }
    }
    printf("v%d S%-2d N %5d K %5d KT %d RW %d: %8.3f ms  %6.2f GB/s weights  %7.1f GFLOPS  max err / sum|wa| %.2e\n",
           ver, parts, N, K, KT, RW, best, wbytes / best / 1e6, 2.0 * N * K * KT / best / 1e6, maxerr);
    return maxerr < 1e-3 ? 0 : 1;
}

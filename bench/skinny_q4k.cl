// Small-batch GEMM Y = A * W^T reading ggml Q4_K weights directly (speculative verification, k = KT <= 8 tokens).
// Model and plan: results/skinny-q4k-model-2026-11-13.txt. Version 1: products of 8 positions summed in FP16 (q = 0..15 exact), everything else FP32
// (as the accurate FP16 GEMM, which sums 32 FP16 products before its FP32 carry).
// W: N rows of K/256 Q4_K blocks (144 B: d, dmin (half), scales[12] (6-bit scale/min x 8), qs[128]), row_bytes apart.
// A: KT x K half, row-major (activations, converted once per batch). Y: KT x N float, row-major.
// One SIMD8 subgroup owns RW weight rows. Per block, two steps h = 0, 1: a 64 B block read (intel_sub_group_block_read2)
// of each row's qs gives lane l the dwords at bytes 64h + 4l and 64h + 32 + 4l, i.e. positions p = 4l .. 4l+3 of the
// byte pairs 2h and 2h+1; low nibbles of pair c are sub-block 2c, high nibbles sub-block 2c+1 (ggml order). So step h
// covers sub-blocks 4h .. 4h+3 at the same positions in every lane (scales are uniform across the subgroup). The
// activation positions of a lane are the same for every row: loaded once per step and token, reused for the RW rows.
// Sum per (row, token, sub-block): d*sc * sum(q*a) - dmin*m * sum(a); sum(a) is row-independent.
// Build-time: KT (1..8), RW (1..8). Needs K % 256 == 0, row_bytes % 16 == 0. Grid: (N / RW) * 8, local 8 * SG.
// No private arrays (kcl has no scratch): accumulators and activations are macro-expanded scalars / vectors.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#ifndef KT
#define KT 4
#endif
#ifndef RW
#define RW 4
#endif

// Lane j computes scale and min of sub-block j (j = 0..7) of a block header hd (16 B: d, dmin, scales[12])
// (ggml get_scale_min_k4); bytes b[i] of the 12 scale bytes start at header byte 4.
inline uint hbyte(uint4 h, uint i) {   // header byte i (0..15), i may differ per lane
    const uint w = i < 4 ? h.s0 : i < 8 ? h.s1 : i < 12 ? h.s2 : h.s3;
    return (w >> (8 * (i & 3))) & 0xFF;
}
inline float2 scale_min_lane(uint4 h, uint j) {
    const uint lo = hbyte(h, 4 + j), hi = hbyte(h, 8 + j), lo4 = hbyte(h, 4 + (j & 3));
    const uint sc = j < 4 ? (lo & 63) : ((hi & 0xF) | ((lo4 >> 6) << 4));
    const uint mn = j < 4 ? (hi & 63) : ((hi >> 4) | ((hbyte(h, 8 + (j & 3)) >> 6) << 4));
    return (float2)((float)sc, (float)mn);
}
// For j >= 4 ggml uses q[j+4] (= header byte 8 + j), q[j-4] (= byte 4 + j - 4) and q[j] (= byte 4 + j): above,
// lo = byte 4 + j = q[j], hi = byte 8 + j = q[j + 4], lo4 = byte 4 + (j - 4) = q[j - 4] for j >= 4.

#define T_LIST(X) X(0) X(1) X(2) X(3) X(4) X(5) X(6) X(7)
#define R_LIST(X) X(0) X(1) X(2) X(3) X(4) X(5) X(6) X(7)
#define ON_T(t, ...) if ((t) < KT) { __VA_ARGS__ }
#define ON_R(r, ...) if ((r) < RW) { __VA_ARGS__ }

#define DECL_ACC_ONE(r, t) float acc_##r##_##t = 0.f;
#define DA0(t) DECL_ACC_ONE(0, t)
#define DA1(t) DECL_ACC_ONE(1, t)
#define DA2(t) DECL_ACC_ONE(2, t)
#define DA3(t) DECL_ACC_ONE(3, t)
#define DA4(t) DECL_ACC_ONE(4, t)
#define DA5(t) DECL_ACC_ONE(5, t)
#define DA6(t) DECL_ACC_ONE(6, t)
#define DA7(t) DECL_ACC_ONE(7, t)
#define DECL_ACC_R(r) T_LIST(DA##r)

// per block and row: d*scale and dmin*min of the lane's sub-block (lane j -> sub-block j)
#define DECL_SC(r) float2 sm_##r = 0;
#define SCALES(r) ON_R(r, { const uint4 hd = *(__global const uint4 *)(W + (size_t)(n0 + r) * row_bytes + (size_t)b * 144); \
    const half2 dd = as_half2(hd.s0); sm_##r = scale_min_lane(hd, lane) * (float2)((float)dd.s0, (float)dd.s1); })

// activations of token t at this lane's positions 4l..4l+3 of sub-blocks 4h..4h+3 (half4 each) and their FP32 sums
#define DECL_A(t) half4 a0_##t = 0, a1_##t = 0, a2_##t = 0, a3_##t = 0; float4 sa_##t = 0;
#define SUM4(v) (((float)v.s0 + (float)v.s1) + ((float)v.s2 + (float)v.s3))
#define LOAD_A(t) ON_T(t, { __global const half *at = ap + (size_t)(t) * K; \
    a0_##t = vload4(0, at); a1_##t = vload4(0, at + 32); a2_##t = vload4(0, at + 64); a3_##t = vload4(0, at + 96); \
    sa_##t = (float4)(SUM4(a0_##t), SUM4(a1_##t), SUM4(a2_##t), SUM4(a3_##t)); })

#define HDOT4(q, a) fma(q.s0, a.s0, fma(q.s1, a.s1, fma(q.s2, a.s2, q.s3 * a.s3)))
#define RT(r, t) ON_T(t, acc_##r##_##t += dv.s0 * (float)HDOT4(q0, a0_##t) + dv.s1 * (float)HDOT4(q1, a1_##t) \
    + dv.s2 * (float)HDOT4(q2, a2_##t) + dv.s3 * (float)HDOT4(q3, a3_##t) - dot(mv, sa_##t);)
#define RT0(t) RT(0, t)
#define RT1(t) RT(1, t)
#define RT2(t) RT(2, t)
#define RT3(t) RT(3, t)
#define RT4(t) RT(4, t)
#define RT5(t) RT(5, t)
#define RT6(t) RT(6, t)
#define RT7(t) RT(7, t)
#define ROW(r) ON_R(r, { \
    const uint2 raw = intel_sub_group_block_read2((__global const uint *)(W + (size_t)(n0 + r) * row_bytes + (size_t)b * 144 + 16 + 64 * h)); \
    const uchar4 b0 = as_uchar4(raw.s0), b1 = as_uchar4(raw.s1); \
    const half4 q0 = convert_half4(b0 & (uchar4)0xF), q1 = convert_half4(b0 >> (uchar4)4); \
    const half4 q2 = convert_half4(b1 & (uchar4)0xF), q3 = convert_half4(b1 >> (uchar4)4); \
    const float4 dv = (float4)(sub_group_broadcast(sm_##r.s0, 4 * h), sub_group_broadcast(sm_##r.s0, 4 * h + 1), \
                               sub_group_broadcast(sm_##r.s0, 4 * h + 2), sub_group_broadcast(sm_##r.s0, 4 * h + 3)); \
    const float4 mv = (float4)(sub_group_broadcast(sm_##r.s1, 4 * h), sub_group_broadcast(sm_##r.s1, 4 * h + 1), \
                               sub_group_broadcast(sm_##r.s1, 4 * h + 2), sub_group_broadcast(sm_##r.s1, 4 * h + 3)); \
    T_LIST(RT##r) })

#define ST(r, t) ON_T(t, { const float v = sub_group_reduce_add(acc_##r##_##t); if (lane == 0) Y[(size_t)(t) * N + n0 + r] = v; })
#define ST0(t) ST(0, t)
#define ST1(t) ST(1, t)
#define ST2(t) ST(2, t)
#define ST3(t) ST(3, t)
#define ST4(t) ST(4, t)
#define ST5(t) ST(5, t)
#define ST6(t) ST(6, t)
#define ST7(t) ST(7, t)
#define STORE_R(r) ON_R(r, T_LIST(ST##r))

__attribute__((intel_reqd_sub_group_size(8)))
__kernel void skinny_q4k(__global const uchar *W, ulong row_bytes, __global const half *A, __global float *Y, int N, int K) {
    const uint lane = get_sub_group_local_id();
    const int n0 = (int)(get_global_id(0) / 8) * RW;
    if (n0 >= N) return;
    R_LIST(DECL_ACC_R)
    R_LIST(DECL_SC)
    T_LIST(DECL_A)
    __attribute__((opencl_unroll_hint(1))) for (int b = 0; b < K / 256; b++) {
        R_LIST(SCALES)
        __attribute__((opencl_unroll_hint(1))) for (int h = 0; h < 2; h++) {
            __global const half *ap = A + 256 * b + 128 * h + 4 * (int)lane;   // sub-block 4h starts at 32 * 4h
            T_LIST(LOAD_A)
            R_LIST(ROW)
        }
    }
    R_LIST(STORE_R)
}
